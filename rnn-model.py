
class RNNModel(nn.Module):
    """
    Neural Network Module with an embedding layer, a recurent module and an output linear layer
    
    Arguments:
        rnn_type(str) -- type of rnn module to use options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']
        input_size(int) -- size of the dictionary of embeddings
        embz_size(int) -- the size of each embedding vector
        hidden_size(int) -- the number of features in the hidden state 
        batch_size(int) -- the size of training batches 
        output_size(int) -- the number of output classes to be predicted
        num_layers(int) -- Number of recurrent layers
        dropout -- dropout probabilty
        bidirectional(boolean) -- If True, becomes a bidirectional RNN
        tie_weights(boolean) -- if True, ties the weights of the embedding and output layer
    
    Returns:
        output of shape (batch_size, output_size) -- tensor containing the sigmoid activation on the 
                                                    output features h_t from the last layer of the rnn, 
                                                    for the last time-step t.
    """
    
    
    def __init__(self, rnn_type, input_size, embz_size, hidden_size, batch_size, output_size,
                num_layers=1, dropout=0.5, bidirectional=True, tie_weights=False):
        super().__init__()
        
        if bidirectional: self.num_directions = 2
        else: self.num_directions = 1

        self.hidden_size, self.output_size, self.embz_size = hidden_size, output_size, embz_size
        self.bidirectional, self.rnn_type, self.num_layers = bidirectional, rnn_type, num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(input_size, embz_size)
        self.output_layer = nn.Linear(hidden_size*self.num_directions, output_size)
        self.init_hidden(batch_size)
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embz_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, bidirectional=bidirectional)
        else:
            try:
                nonlinearity = {'RNN_TANH':'tanh', 'RNN_RELU':'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for '--rnn_type' was supplied,
                                    options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']""")
            self.rnn = nn.RNN(embz_size, hidden_size, num_layers=num_layers, 
                           dropout=dropout, bidirectional=bidirectional, nonlinearity=nonlinearity)    
            
        if tie_weights:
            if hidden_size != embz_size:
                raise ValueError("When using the tied flag, hidden size must be equal to embeddign size")
            elif bidirectional:
                raise ValueError("When using the tied flag, set bidirectional=False")
            self.output_layer.weight = self.embedding_layer.weight    
                    
    def init_emb_weights(self, vector_weight_matrix):
        self.embedding_layer.weight.data.copy_(vector_weight_matrix)
        
    def init_identity_weights(self):
        if self.rnn_type == 'RNN_RELU':
            self.rnn.weight_ih_l0.data.copy_(torch.eye(self.hidden_size, self.embz_size))
            self.rnn.weight_hh_l0.data.copy_(torch.eye(self.hidden_size, self.hidden_size))

            if self.bidirectional:
                self.rnn.weight_ih_l0_reverse.data.copy_(torch.eye(self.hidden_size, self.embz_size))
                self.rnn.weight_hh_l0_reverse.data.copy_(torch.eye(self.hidden_size, self.hidden_size))  
        else:
            pass
    
    def init_hidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            self.hidden = (V(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)),
                           V(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)))
        else:
            self.hidden = V(torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))
                
    def forward(self, seq):
        batch_size = seq[0].size(0)
        if self.hidden[0].size(1) != batch_size:
            self.init_hidden(batch_size)
        input_tensor = self.drop(self.embedding_layer(seq))
        output, hidden = self.rnn(input_tensor, self.hidden)
        self.hidden = repackage_var(hidden)
        output = self.drop(self.output_layer(output))
        return F.sigmoid(output[-1, :, :])

