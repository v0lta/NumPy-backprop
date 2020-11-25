Train LSTM-, GRU- and Elman-RNN cells in NumPy
----------------------------------------------

Type i.e. ```python train_adding.py``` to train an LSTM cell to solve the adding problem in NumPy. It takes a while to converge,
as illustrated in the plot below.

Similarly ```python train_memory.py``` can be used to start training
on the memory problem. 

For both problems, GRUs can be trained by adding ```--cell_type```. 
For example ```python train_memory.py --cell_type GRU``` will train a Gated recurrent unit on the memory problem.
For all possible arguments type 
```python train_memory.py -h```. 

The models will take a while to converge as illustrated below on the adding problem.

![alt text](loss_adding_lstm.png "LSTM-Adding")
