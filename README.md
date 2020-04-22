# [Forecasting directional movements of stock-prices for intraday trading using LSTM and random-forest](https://arxiv.org/abs/2004.10178)

**https://arxiv.org/abs/2004.10178** <br>
**Pushpendu Ghosh, Ariel Neufeld, Jajati K Sahoo**

We employ both random forests on the one hand and LSTM networks (more precisely CuDNNLSTM) on the other hand as training methodology to analyze their effectiveness in forecasting out-of-sample directional movements of constituent stocks of the S&amp;P 500, for intraday trading, from January 1993 till December 2018.

#### Requirements
```
pip install scikit-learn==0.20.4
pip install tensorflow==1.14.0
```

## Plots
We plot three important metrics to quantify the effectiveness of our model: [Intraday-240,3-LSTM.py](Intraday-240%2C3-LSTM.py) and [Intraday-240,3-RF.py](Intraday-240%2C3-RF.py), in the period January 1993 till December 2018. <br>
**Intraday LSTM**: [Intraday-240,3-LSTM.py](Intraday-240%2C3-LSTM.py) <br>
**Intraday RF**: [Intraday-240,3-RF.py](Intraday-240%2C3-RF.py) <br>
**Next Day LSTM, krauss18**: [NextDay-240,1-LSTM.py](NextDay-240%2C1-LSTM.py) [1] <br>
**Next Day RF, krauss17**: [NextDay-240,1-RF.py](NextDay-240%2C1-RF.py) [2] <br>

#### Cumulative Money growth (after transaction cost)
<div>
<img src="result-images/money1.jpg" width="32.5%">
<img src="result-images/money2.jpg" width="32.5%">
<img src="result-images/money3.jpg" width="32.5%">
</div>

#### Average daily returns (after transaction cost)
<div>
<img src="result-images/barRet1.jpg" width="32.5%">
<img src="result-images/barRet2.jpg" width="32.5%">
<img src="result-images/barRet3.jpg" width="32.5%">
</div>

#### Average (Annualized) Sharpe ratio (after transaction cost)
<div>
<img src="result-images/barSharpe1.jpg" width="32.5%">
<img src="result-images/barSharpe2.jpg" width="32.5%">
<img src="result-images/barSharpe3.jpg" width="32.5%">
</div>

## Acknowledgements
The first author gratefully acknowledges the NTU-India Connect Research Internship Programme which allowed him to carry out part of this research project while visiting the Nanyang Technological University, Singapore.<br>
The second author gratefully acknowledges financial support by the NAP Grant.

## References
[1] [Fischer, Thomas, and Christopher Krauss. "Deep learning with long short-term memory networks for financial market predictions." European Journal of Operational Research 270.2 (2018): 654-669.](https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf) <br>
[2] [Krauss, Christopher, Xuan Anh Do, and Nicolas Huck. "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." European Journal of Operational Research 259.2 (2017): 689-702.](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)


