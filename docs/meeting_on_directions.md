# Meeting on Directions in ASR and Stuttering
The Frontier’s work established initial estimated of word error rates (WER) in people who stutter (PWS). This work, based 15 recordings from 13 PWS yielded an average WER estimate of 24.5%. While this provides the first estimates of ASR performance on stuttered speech and evidences systematic biases towards the fluent, typical speaking population, there is a lack of breadth and depth of the analyses.

The following is proposed to address this gap in the knowledge field:

1. Estimation of WER in PWS using a.) more data and b.) co-dependency analyses of the ASR’s error and the speech state.
2. Methods and protocols to improve ASR systems
3. Miscellaneous additional applications

## In-depth analysis of WER of stuttering from machine learning models
### Increasing dataset size
To improve generalisability of these findings we need to expand the dataset size. From UCLASS, there are only 15 recordings that have word level transcriptions to use as reference. Additionally we have a collection of privately available audio recordings in the Speech Lab UCL (Herein referred to as Speech Lab Archive of Stuttered Speech or SLASS). While this archive has approximately 1,000 audio recordings of PWS (variable length), the number with word-level transcriptions is known (Work item 1.).\
There are additional stuttered speech corpora such as, FluencyBank (Ratner & MacWhinney, 2018), SEP-28k (Lea et al., 2021) and KSoF (Bayerl et al., 2022). All have their constraints. FluencyBank provides sentence level transcriptions only and has a very limited number of recordings. SEP-28k have that most amount of freely available data but it is not transcribed (see possible uses later). Finally, KSoF also does not provide transcriptions but, is in German (see possible uses later).

### Increasing analysis depth
To improve depth of analysis and understanding, we will investigate how WER varies as a function of stutter type & whether the type of error is co-dependent (substitution, deletion, insertion).\
We will need to decide on the appropriate method to estimate co-dependency (Work item 2.). Ideally we would have a method that provides us with: given that the word was a PWR, what is the probability that the transcript provides a an error and of what type? This can be extended and perhaps even the inverse can be made true to help detect stuttering. I.e., there is a low certainty to this word transcription, what is the probability of prolongation or indeed any stutter type?\
For this we will need to create a co-dependency table (Work item 3.) between the word, its error type (substitution, deletion, insertion or none) and its fluency type (fluent, prolongation, part-word repetition or PWS, whole-word repetition or WWR and, block).

### Increasing model diversity
To improve applicability of findings, we will test a range of models. As a starting point, testing with Whisper (Radford et al., 2022), wav2vec 2.0 (Baevski et al., 2020), Google USM (Zhang et al., 2023) and, HuBERT (Hsu et al., 2021).

### Increasing Language diversity
To further improve generalisability, we should consider stuttering across multiple languages. Currently there are only English speech corpora from PWS with word level transcription. However, KSoF provides openly available speech from German PWS. The KSoF group may have transcriptions (Work item 4.) or we may have to hire someone to transcribe the audio (A student at Düsseldorf for ex.).

## Methods and protocols to improve ASR systems
Better experimentation with temperature and other hyper params is required to provide grounded suggestions for ASR improvements (Work item 5). We considered only temperature and tested this in an exploratory way. We need to be more systematic with our analyses This can be used for guidelines on how to better implement ASR models for stuttered speech as well as perhaps information about how ASR models encode speech in the first place. I.e., using break points to assess how transformers work.

Effect on language auto-detection and possible resolutions. (I found the bizarre occurrence of stuttered English speech being classified as Welsh and Spanish by Whisper!). If we can document this and figure out why this occurs this could be a significant improvement in current ASR models.

## Misc. Additional Applications
Application of (optimised for stuttered speech) ASR to interval datasets to provide “soft” labels and probability estimates of where the stutter occurs across thee 3-sec interval (Work item 6.). This should already be done for the German data if un-transcribed.

Performing these analyses on other language corpora and re-assessing - we could approach Kassel group to run an English/German collaboration? Or perhaps Clarissa for German level transcript checking.

## Work items

- [ ] Number of audio records with appropriate word-level transcription (L.B.).
	- [ ] Quality check of audio transcripts from out-of-UCL databases.
	- [ ] Cross-compare against ASR transcriptions to get an initial guess at what databases are of sufficient quality (Using WER to check quality of recording and the “true” transcriptions).
- [ ] Methods to estimate co-dependency (L.B.).
	- [ ] Look into  Martin & Tang’s ([2020](https://slam.phil.hhu.de/papers/2020/MartinTang_2020_HabitualBe_Interspeech.pdf)) analysis method - looking at probability of a mis-transcription given a stutter as well as mis-transcriptions around the target.
	- [ ] Create co-dependency table between WER and fluency state (L.B.).
- [ ] Contact Kassel for transcripts (L.B.) and failing that, set up transcription with German student (K.T.).
- [ ] Better experimentation with model hyper-params (L.B. & K.T.).
- [ ] Apply automatic transcriptions to German audio data (L.B. & K.T.)

## References
Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33, 12449-12460.

Bayerl, S. P., von Gudenberg, A. W., Hönig, F., Nöth, E., & Riedhammer, K. (2022). KSoF: The Kassel State of Fluency Dataset--A Therapy Centered Dataset of Stuttering. arXiv preprint arXiv:2203.05383.

Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29, 3451-3460.

Lea, C., Mitra, V., Joshi, A., Kajarekar, S., & Bigham, J. P. (2021, June). Sep-28k: A dataset for stuttering event detection from podcasts with people who stutter. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6798-6802). IEEE.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023, July). Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning (pp. 28492-28518). PMLR.

Ratner, N. B., & MacWhinney, B. (2018). Fluency Bank: A new resource for fluency research and practice. Journal of fluency disorders, 56, 69-80.

Zhang, Y., Han, W., Qin, J., Wang, Y., Bapna, A., Chen, Z., ... & Wu, Y. (2023). Google usm: Scaling automatic speech recognition beyond 100 languages. arXiv preprint arXiv:2303.01037.