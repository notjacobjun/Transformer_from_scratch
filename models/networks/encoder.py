class Encoder:
	def __init__(self, num_heads, max_len, device, d_model, num_FFN, ffn_hidden, n_layers, drop_prob, encoder_voc_size):
		self.num_heads = num_heads
		self.num_FFN = num_FFN
		self.max_len = max_len
		self.device = device
		self.d_model = d_model
		self.ffn_hidden = ffn_hidden
		self.n_layers = n_layers
		self.drop_prob = drop_prob
		self.encoder_voc_size = encoder_voc_size

	def forward(self, x, s_mask):
		# parse the input using some util function and tokenize it
