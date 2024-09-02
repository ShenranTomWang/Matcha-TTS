# Important Changes For Multilingual Experiments:
Anywhere there is speaker embedding, there should be language embedding.
1. [cleaners updated](https://github.com/ShenranTomWang/Matcha-TTS/commit/323bfafb16dc4701a030ff3d310efd9ab421cf9a)
2. config files: data should have n_lang config option, model should have lang_emb_dim config option. [`configs/model/multilingual.yaml`](configs/model/multilingual.yaml) and [`configs/experiment/multilingual.yaml`](configs/experiment/multilingual.yaml) are added for multilingual experiment.
3. In file [`text_mel_datamodule.py`](matcha/data/text_mel_datamodule.py): added language embedding for `TextMelDataModule`, `TextMelDataModule.train_dataloder()`, `TextMelDataModule.val_dataloder()`, `TextMelDataset`, `TextMelDataset.get_datapoint()`, `TextMelBatchCollate` and `TextMelBatchCollate.__call__()`. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/fa5768387762f95199d86d67d93c5551bbb9d172#diff-1135090fccfa8e6303de7c2a6158fa5c278856fa086020860167d1d6dd34c9e5) for details. Bugfix added in [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/d05c1a8e7739006976606786e21755a88eafe493) to properly parse language and speaker embedding from file.
4. In file [`decoder.py`](matcha/models/components/decoder.py): added language embedding for `Decoder.forward()`. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/fa5768387762f95199d86d67d93c5551bbb9d172#diff-1135090fccfa8e6303de7c2a6158fa5c278856fa086020860167d1d6dd34c9e5) for details.
5. In file [`flow_matching.py`](matcha/models/components/flow_matching.py): added language embedding for `BASECFM`, `BASECFM.forward()`, `BASECFM.solve_euler()`, `BASECFM.compute_loss()` and added dimensions for `CFM` conditional on `n_lang`. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/fa5768387762f95199d86d67d93c5551bbb9d172#diff-1135090fccfa8e6303de7c2a6158fa5c278856fa086020860167d1d6dd34c9e5) for details.
6. In file [`text_encoder.py`](matcha/models/components/text_encoder.py), added language embedding for `TextEncoder` and `TextEncoder.forward()`. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/fa5768387762f95199d86d67d93c5551bbb9d172#diff-1135090fccfa8e6303de7c2a6158fa5c278856fa086020860167d1d6dd34c9e5) for details.
7. In file [`matcha_tts.py`](matcha/models/matcha_tts.py), added language embedding for `MatchaTTS`, `MatchaTTS.synthesise()` and `MatchaTTS.forward()`. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/fa5768387762f95199d86d67d93c5551bbb9d172#diff-1135090fccfa8e6303de7c2a6158fa5c278856fa086020860167d1d6dd34c9e5) for details. Bugfix added in [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/69b75f83a36838aaa206cf50b86c190da604c1b0#diff-3f488ba92fcd103f2d2ff1670718a031bc007845fc2efdaf9fcb92c5c4d36bee) to accept language embedding in `MatchaTTS.forward()`, [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/6a74c205f1b3221524dd9856dcb94e6ae93bbf82) documentation update and syntax fix, [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/83dc48afd74d49710498c048019ae3b5adb3daa7) passing parameters correctly in `MatchaTTS.__init__()`
8. In file [`baselightningmodule.py`](matcha/models/baselightningmodule.py), passing language embedding to model. See [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/69b75f83a36838aaa206cf50b86c190da604c1b0#diff-4d0df8e93d84023bb4562fe91e5dda0df742e7ebfdfa1acbdceffc49dd96dc8f) for details. Bugfix added in [commit](https://github.com/ShenranTomWang/Matcha-TTS/commit/38a454d73e8dd9dadedd45d3973a9f3194ff7cc3) to accept language embedding in `BaseLightningClass.on_validation_end()`.