# Dataset

PaySim1, a simulated mobile money dataset: 6,362,620 transactions over 30
simulated days, 8,213 of them fraud (0.129%).

The CSV is around 490 MB so it isn't committed. Download it from
<https://www.kaggle.com/datasets/ealaxi/paysim1>, put
`PS_20174392719_1491204439457_log.csv` in this folder, then:

```bash
cd backend && python train.py
```

You only need this if you want to retrain. The trained model is already in
`backend/model/`.

## Worth knowing before you trust any result on this data

PaySim creates a fraudulent transaction by emptying the sender's account, so
97.7% of fraud rows have `oldbalanceOrg == amount` and `newbalanceOrig == 0`,
and no legitimate row does. You can separate the classes with a two line if
statement, and any model given features built on that will look near perfect
without having learned anything.

`train.py` reports that rule as a baseline next to the model, and refits
without those features so the difference is visible. Both numbers are in the
results section of the main README.
