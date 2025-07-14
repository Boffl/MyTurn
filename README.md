# Do Speakers Mark the Beginning of an Utterance using non-linguistic Cues?
No evidence found for this...

## Scripts
### `data_prep.py`
Extract and pool MFCCs from audio. Create a dataset of stimuli, each is a pairs of the MFCCs of two audio snippets combined
- Takes as positional argument the path to a directory with .wav files (required).
- Optionally you can add noise to the data by specifying the stdev of Gaussian noise to be applied before extracting the MFCCs. Defaults to no noise.

**Example Call**
```
python .\scripts\data_utils.py .\07_stimuli\ --noise 10
```

**Outputs**
- `MFCCs{\_noise\_10}` Directory with the MFCC files for each .wav file in the audio directory. If noise was added it is mentioned in the directory name, with the defined stdev
- `feature\_dataset{\_noise\_10}.pickle` Python dictionary with a key for each condition ('b', 'm', 'e'). The values are the X and y vectors, where X are the concatenated MFCCs for the stimulus pair and y is the binary "same" vs. "different" class ("same" is the target class)

### `analysis.py`
Run a random forest classifier based on pairs of audio. Classifies into classes *same-speaker* and *different-speaker*. 
- Expects outputs from `data_prep.py` in the directory where it is run. 
- No positional arguments. Optional arguments:
    - `--noise`: Same as above, defaults to no noise.
    - `--folds`: Number of folds for cross evaluation, defaults to 5.
    - `--all_data`: Run analysis on the complete dataset and not for each subset (beginning, middle and end). Defaults to **False**.

**Example Call**
```
python .\scripts\analysis.py --all_data
```

**outputs**
Writes to the `results` directory. 
- A file called `allClasses*.csv` with a classification report with average scores (precision, recall and f1), including both classes as well as weighted and macro average. 
- A file called `onlyTarget*.csv` with results for each fold, only considering *same-speaker* as the target class.
- If `--all_data` is set to **False**, it will output two .csv files for each postition.
