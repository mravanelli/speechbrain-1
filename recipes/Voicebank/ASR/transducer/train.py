# /usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and CTC loss on Voicebank

To run this recipe, do the following:
> python train.py hparams/{hyperparameter file} --data_folder /path/to/noisy-vctk

Use your own hyperparameter file or the provided `hyperparams.yaml`

To use noisy inputs, change `input_type` field from `clean_wav` to `noisy_wav`.
To use pretrained model, enter the path in `pretrained` field.

Authors
 * Peter Plantinga 2020
"""
import os
import sys
import torch
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml


# Define training procedure
class ASR_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns, phn_lens = batch.phn_encoded

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            phns = self.hparams.wav_augment.replicate_labels(phns)

        # Model computations
        feats = self.hparams.compute_features(wavs)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
        
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)
        x = self.modules.enc_lin(x)

        # Prepend bos token at the beginning
        y_in = sb.dataio.dataio.prepend_bos_token(
            phns, self.hparams.blank_index
        )
        e_in = self.modules.emb(y_in)
        h, _ = self.modules.dec(e_in)
        h = self.modules.dec_lin(h)

        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # output layer for seq2seq log-probabilities
        logits = self.modules.output(joint)

        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.Greedysearcher(x)
            return logits, wav_lens, hyps

        elif stage == sb.Stage.TEST:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(x)
            return logits, wav_lens, best_hyps
        return logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the loss."
        ids = batch.id
        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)
        
        if hasattr(self.hparams, "fea_augment"):
                phns = self.hparams.fea_augment.replicate_labels(phns)
                phn_lens = self.hparams.fea_augment.replicate_labels(
                    phn_lens
                )

        if stage == sb.Stage.TRAIN:
            predictions, wav_lens = predictions
        else:
            predictions, wav_lens, hyps = predictions

        # Transducer loss use logits from RNN-T model.
        loss = self.hparams.compute_cost(predictions, phns, wav_lens, phn_lens)
        self.transducer_metrics.append(
            ids, predictions, phns, wav_lens, phn_lens
        )

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.label_encoder.decode_ndim
            )

        return loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.transducer_metrics = self.hparams.transducer_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "PER": per},
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per}, min_keys=["PER"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            run_on_main(
                save_metrics_to_file,
                args=[
                    self.hparams.test_wer_file,
                    self.transducer_metrics,
                    self.per_metrics,
                ],
            )


def save_metrics_to_file(wer_file, transducer_metrics, per_metrics):
    with open(wer_file, "w") as w:
        w.write("Transducer loss stats:\n")
        transducer_metrics.write_stats(w)
        w.write("\nPER stats:\n")
        per_metrics.write_stats(w)
        print(
            "Transducer and PER stats written to file",
            hparams["test_wer_file"],
        )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 1. Define audio pipeline:
    @sb.utils.data_pipeline.takes(hparams["input_type"])
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 2. Define text pipeline:
    @sb.utils.data_pipeline.takes("phones")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phones):
        phn_list = phones.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    # 3. Create datasets
    data = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "phn_encoded"],
        )

    # Sort train dataset and ensure it doesn't get un-sorted
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        data["train"] = data["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending",
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    # 4. Fit encoder:
    # Load or compute the label encoder (with multi-gpu dpp support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[data["train"]],
        output_key="phn_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    return data, label_encoder


# Begin Recipe!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Prepare data on one process
    from voicebank_prepare import prepare_voicebank  # noqa E402

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )
    if "prepare_noise_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    
    if "prepare_rir_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    datasets, label_encoder = dataio_prep(hparams)

    # Load pretrained model
    if "pretrained" in hparams:
        state_dict = torch.load(hparams["pretrained"])
        hparams["modules"]["model"].load_state_dict(state_dict)

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        run_opts=run_opts,
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Fit the data
    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Test the checkpoint that does best on validation data (lowest PER)
    asr_brain.evaluate(
        datasets["test"],
        min_key="PER",
        test_loader_kwargs=hparams["dataloader_options"],
    )