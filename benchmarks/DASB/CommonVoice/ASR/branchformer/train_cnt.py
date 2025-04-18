#!/usr/bin/env/python3
""" Script for training an ASR model evaluating an SSL representation
model on one language from the CommonVoice dataset. A SentencePiece tokenizer
with number of tokens equal to <output_neurons> is learned in a first phase, on
the considered language.

Authors
 * Pooneh Mousavi 2024
"""

import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
import torchaudio
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
import time
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        #p_tokens, _ = batch.speech_tokens
        tokens_bos, _ = batch.tokens_bos

        #embeddings = self.modules.discrete_embedding_layer(p_tokens)
        #if  hasattr(self.hparams,'embedding_strg') and  self.hparams.embedding_strg == 'concat':
        #    B, T, N_Q, D = embeddings.shape
        #    #feats = embeddings.view(B,T,N_Q *D)
        #    feats = embeddings.reshape(B, T, N_Q * D)

        #else:
        #    att_w = self.modules.attention_mlp(embeddings)  # [B, T, N-Q, 1]
        #    feats = torch.matmul(att_w.transpose(2, -1), embeddings).squeeze(
        #        -2
        #    )  # [B, T, D]
        feats = self.modules.weighted_ssl_model(
            wavs
        )

        p_seq = None

        if type(self.modules.encoder).__name__ == "VanillaNN":
            enc_out = self.modules.encoder(feats)

        elif type(self.modules.encoder).__name__ == "LSTM":
            enc_out, _ = self.modules.encoder(feats)

        elif type(self.modules.encoder).__name__ == "TransformerASR":
            enc_out, pred, _, _ = self.modules.encoder(feats, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index)
            pred = self.modules.seq_lin(pred)
            p_seq = self.hparams.log_softmax(pred)
            p_seq = p_seq
        else:
            raise NotImplementedError

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        if  stage == sb.Stage.VALID and current_epoch % self.hparams.valid_search_interval == 0:
            if type(self.modules.encoder).__name__ == "TransformerASR":
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), wav_lens
            )
            else:
                hyps = sb.decoders.ctc_greedy_decode(
                    p_ctc, wav_lens, blank_id=self.hparams.blank_index
                )

        elif stage == sb.Stage.TEST:
            if type(self.modules.encoder).__name__ == "TransformerASR":
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), wav_lens
                )
        
        return p_ctc,p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc,p_seq, wav_lens, hyps = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens
        tokens_eos, tokens_eos_lens = batch.tokens_eos

         # Label Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss_seq = self.hparams.seq_cost(
                p_seq, tokens_eos, length=tokens_eos_lens
            ).sum()

        loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens
            ).sum()

        loss = (self.hparams.ctc_weight * loss_ctc
                + (1 - self.hparams.ctc_weight) * loss_seq
            )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                if type(self.modules.encoder).__name__ == "TransformerASR":
                # Decode token terms to words
                    predicted_words = [
                        tokenizer.sp.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                    ]
                else:
                    if stage == sb.Stage.VALID:
                        # Decode token terms to words
                        predicted_words = self.tokenizer(
                            hyps, task="decode_from_list"
                        )
                    elif stage == sb.Stage.TEST:
                        predicted_words = [
                            hyp[0].text.split(" ") for hyp in hyps
                        ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            if type(self.hparams.scheduler).__name__ == "NewBobScheduler":
                lr, new_lr = self.hparams.scheduler(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            elif type(self.hparams.scheduler).__name__ == "NoamScheduler":
                lr = self.hparams.scheduler.current_lr
            else:
                raise NotImplementedError

            optimizer = self.optimizer.__class__.__name__
            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                min_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    #def init_optimizers(self):
    #    "Initializes the weights optimizer and model optimizer"
    #    self.model_optimizer = self.hparams.model_opt_class(
    ##        self.hparams.model.parameters()
    #    )
    #    self.optimizers_dict = {
    #        "model_optimizer": self.model_optimizer,
    #    }
    #    # Initializing the weights
    #    if self.checkpointer is not None:
    #        self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": data_folder},
    )

    # We also sort the validation data so it is faster to validate
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    # 1. Define tokens pipeline:
    #tokens_loader = hparams["tokens_loader"]
    #num_codebooks = hparams["num_codebooks"]

    #@sb.utils.data_pipeline.takes("id")
    #@sb.utils.data_pipeline.provides("speech_tokens")
    #def tokens_pipeline(id):
    
    #    tokens = tokens_loader.tokens_by_uttid(id, num_codebooks=num_codebooks)
    #    return tokens

    #sb.dataio.dataset.add_dynamic_item(datasets, tokens_pipeline)

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(sig)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation
    from common_voice_prepare import prepare_common_voice  # noqa

    # multi-gpu (ddp) save data preparation
    # Due to DDP, we do the preparation ONLY on the main python process
    run_on_main(
        prepare_common_voice,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "train_tsv_file": hparams["train_tsv_file"],
            "dev_tsv_file": hparams["dev_tsv_file"],
            "test_tsv_file": hparams["test_tsv_file"],
            "accented_letters": hparams["accented_letters"],
            "language": hparams["language"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],  # Number of considered tokens
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    # Use pretrained embeddings
    if hparams["pretrain_embeddings"]:
        tokens_loader = hparams["tokens_loader"]
        embs = tokens_loader.load_pretrained_embeddings(
            hparams["pretain_embeddings_folder"]
        )
        if isinstance(hparams["num_codebooks"], int):
            embs = embs[: hparams["num_codebooks"] * hparams["vocab_size"],]
        # For discrete SSL, num_codebooks is a list used to determine which layers to use.
        # It is not sequential and can be, for example, [0, 1] or [1, 4].
        elif isinstance(hparams["num_codebooks"], list):
            indices = [
                i
                for codebook_idx in hparams["num_codebooks"]
                for i in range(
                    codebook_idx * hparams["vocab_size"],
                    (codebook_idx + 1) * hparams["vocab_size"],
                )
            ]
            indices = torch.tensor(indices, dtype=torch.long)
            embs = embs[indices]
        hparams["discrete_embedding_layer"].init_embedding(embs)

    # Log number of parameters/buffers
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["model_opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer
    vocab_list = [
        tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())
    ]

    #from speechbrain.decoders.ctc import CTCBeamSearcher

    #test_searcher = CTCBeamSearcher(
    #    **hparams["test_beam_search"],
    #    vocab_list=vocab_list,
    #)

    # Training
    start_time = time.time()  # Start the timer
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    end_time = time.time()  # End the timer
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    logger.info(f"Model execution time: {elapsed_time:.6f} seconds")

    # Testing
    if hparams["testing"]:
        # Testing
        if not os.path.exists(hparams["output_wer_folder"]):
            os.makedirs(hparams["output_wer_folder"])

        
            asr_brain.hparams.output_wer_folder = os.path.join(
                hparams["output_wer_folder"], f"wer_test.txt"
            )
            asr_brain.evaluate(
                test_data,
                test_loader_kwargs=hparams["test_dataloader_opts"],
                min_key="ACC",
            )