# Do Speakers Mark the Beginning of an Utterance using non-linguistic Cues?
No evidence found for this...

## Scripts
### `data_prep.py`
Extract and pool MFCCs from audio. Create a dataset of stimuli, each is a pairs of the MFCCs of two audio snippets combined
- Takes as positional argument the path to a directory with .wav files (required)
- Optionally you can add noise to the data by specifying the stdev of Gaussian noise to be applied before extracting the MFCCs. Defaults to no noise

**Example Call**
```
python .\scripts\data_utils.py .\07_stimuli\ 10
```

**Outputs**
- `MFCCs{\_noise\_10}` Directory with the MFCC files for each .wav file in the audio directory. If noise was added it is mentioned in the directory name, with the defined stdev
- `feature\_dataset{\_noise\_10}.pickle` Python dictionary with a key for each condition ('b', 'm', 'e'). The values are the X and y vectors, where X are the concatenated MFCCs for the stimulus pair and y is the binary "same" vs. "different" class ("same" is the target class)
