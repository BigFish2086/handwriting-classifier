# handwriting-classifier

A male/female classifier for handwriting


## 🚩 Usage
1. Clone the repo `git clone https://github.com/BigFish2086/handwriting-classifier`
2. Go into the repo folder `cd handwriting-classifier/src`
3. It's recommended to use a virtual environment
  - using pip
   ```
   python3 -m pip install virtualenv
   virtualenv -p python3 __venv__
   ```
4. Insall the required Libraries `python3 -m pip install -r ../requirements.txt`
5. Start by running `python3 ./setup.py -b` to build the required directory structure
6. Move your `data-set` folder in the current `src` folder to replcae the created one
7. Do the previous step for the `test` folder (if having one to use the model to predict them)
8. Run `python3 ./main -pegc` (or run `python3 ./main -h` to get a help menu)
9. To use the trained models run for example `python3 ./predict.py -i ./test -o ./out -g` \
   (or run `python3 ./predict.py -h` for a help menu as well)
10. To evaulte those predictions done in the previous step `python3 ./evalute.py` \
    (make sure to have a `ground_truth.txt` file to comapre the predictions results with it)
    
#### Note:
- check [Project Directory Structure](./work.md#project-dirictory-structure) in case having any issues

