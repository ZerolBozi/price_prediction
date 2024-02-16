# price_prediction
 
My school graduation project

Predicting financial market prices using LSTM

### conda environment
```
conda env create -f environment.yml
```

### use train script and save it (if 'save_model' is False, it's will save checkpoint)
```
python train_script.py --ticker 2330 --model_name 2330_test --save_model True
```

### fine-tune model
```
python train_script.py --ticker 2330 --checkpoint_name 2330_test
```

### use model
```
python use_model.py --ticker 2330 --model_name lstm_stock_crypto --time_window_size 1
```
