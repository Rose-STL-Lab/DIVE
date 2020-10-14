import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

class AppearanceEncoder(nn.Module):
  '''
  Encode the appearance hidden vectors.
  '''
  def __init__(self, input_size, hidden_size, stat_app_size, var_app_size, n_frames_output, num_layers=2):
    super(AppearanceEncoder, self).__init__()

    self.encoder = nn.LSTM(input_size=input_size + hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
    self.predictor = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)

    self.stat_app_layer = nn.Linear(hidden_size, stat_app_size)

    self.var_app_layer = nn.Linear(hidden_size, var_app_size)
    self.var_app_layer_0 = nn.Linear(hidden_size + stat_app_size, hidden_size)

    self.var_app_ini_layer = nn.Linear(hidden_size, var_app_size)
    self.var_app_ini_layer_0 = nn.Linear(input_size + stat_app_size, hidden_size)

    self.lrelu = nn.LeakyReLU()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.n_frames_output = n_frames_output

  def accumulate_varying_appearance(self, ini_cond, var_app):
    var_app_output = ini_cond
    var_app_outputs = []
    for t in range(var_app.shape[1]):
      var_app_output = var_app_output + var_app[:,t:t+1]
      var_app_outputs.append(var_app_output)
    var_app_output = torch.cat(var_app_outputs, dim=1)
    return var_app_output

  def forward(self, input, prev_obj_hidden):
    '''
    input: temporal factor of size                        (batch_size, n_frames, input_size)
    prev_obj_hidden: hidden state of the previous object  (batch_size, n_frames, hidden_size)
    stat_app_output: static appearance                    (batch_size, n_frames, output_size)
    var_app_output: varying appearance in time            (batch_size, n_frames, output_size)
    '''

    # Encoder takes the aligned input + the hidden state of the previous object.
    input_rnn = torch.cat([input, prev_obj_hidden], dim=-1)
    encoder_output, hidden = self.encoder(input_rnn)
    last_output = encoder_output[:, -1, :]
    stat_app_output = self.stat_app_layer(last_output)
    prev_obj_hidden = encoder_output

    pred_outputs = []
    pred_rnn_input = encoder_output[:, -1:]
    for t in range(self.n_frames_output):
      pred_encoder_output, pred_hidden = self.predictor(pred_rnn_input)
      pred_rnn_input = pred_encoder_output
      pred_outputs.append(pred_encoder_output)
    pred_output = torch.cat(pred_outputs, dim=1)

    initial_conditions = self.var_app_ini_layer(self.lrelu(self.var_app_ini_layer_0(
      torch.cat([input[:,:1], stat_app_output.unsqueeze(1)], dim=-1))))

    encoder_output = torch.cat([encoder_output, pred_output], dim=1)

    var_app_output = self.var_app_layer(self.lrelu(self.var_app_layer_0(
      torch.cat([encoder_output,
                 stat_app_output.unsqueeze(1).repeat(1,encoder_output.size(1),1)], dim=-1))))

    var_app_output = self.accumulate_varying_appearance(initial_conditions, var_app_output)

    return stat_app_output, var_app_output, prev_obj_hidden