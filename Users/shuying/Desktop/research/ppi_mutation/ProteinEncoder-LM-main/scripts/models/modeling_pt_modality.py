from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Callable, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch as pyg_Batch

from .modeling_pt_graph import graph_MPNN
from .modeling_utils import BaseConfig, BaseModel, LayerNorm
from .modeling_pt_bert import BertModel
from .modeling_pt_resnet import ResNet1DInputWithOptAttention, ResNet2DInputWithOptAttention, structureMultiTask_head

# for registering task and model
from mapping import registry


class StructureMultiTaskConfig(BaseConfig):
    """Configuration class for structureMultiTask model
    
    Arguments:
        vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Bert encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Bert encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Bert encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids` passed into `BertModel`.
        
        initializer_range: The sttdev of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps: The epsilon used by LayerNorm.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """
    def __init__(self,
                 vocab_size: int = 28,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 1024,
                 hidden_act: Union[str,Callable] = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 8096,
                 type_vocab_size: int = 2,                 
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 gradient_checkpointing: bool = False,
                 freeze_bert_firstN: int = 0,
                 
                 mp_layer_name: str = None,
                 node_feat_channels: int = 128,
                 edge_feat_channels: int = 128,
                 graph_node_pos_size: int = 1,
                 graph_edge_pos_size: int = 1,
                 mp_atten_heads: int = 4,
                 mp_dropout_rate: float = 0.1,
                 mp_atten_dropout_rate: float = 0.1,
                 add_self_loop: bool = True,
                 mp_num_layers: int = 4,
                 mp_activ_fn: Union[str,Callable] = "gelu",
                 mp_residual: bool = True,
                 mp_return_atten_weights: bool = True,
                 mp_edge_feat_atten: str = 't2s',

                 num_interact_transf_atten: int = 4,
                 seq_max_length: int = 1000, # interact_transfer_attention seq dim

                 num_chunks: int = 4,
                 init_channels: int = 128,
                 num_channels: int = 128,
                 use_region_atten: bool = False,
                 region_atten_n_head: int = 4,
                 interact_activ_fn: Union[str,Callable] = "gelu",
                 interact_dropout: float = 0.1,

                 num_aa_classes: int = 29,
                 num_ss_classes: int = 3,
                 num_rsa_classes: int = 2,
                 num_dist_classes: int = 64,
                 aa_loss_weight: float = 1.0,
                 ss_loss_weight: float = 1.0,
                 rsa_loss_weight: float = 1.0,
                 dist_loss_weight: float = 1.0,
                 use_class_weights: bool = False,

                 **kwargs):
        super().__init__(**kwargs)
        ## config for bert ##
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_bert_firstN = freeze_bert_firstN

        ## config for graph ##
        self.mp_layer_name = mp_layer_name
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.graph_node_pos_size = graph_node_pos_size
        self.graph_edge_pos_size = graph_edge_pos_size
        self.mp_atten_heads = mp_atten_heads
        self.mp_dropout_rate = mp_dropout_rate
        self.mp_atten_dropout_rate = mp_atten_dropout_rate
        self.add_self_loop = add_self_loop
        self.mp_num_layers = mp_num_layers
        self.mp_activ_fn = mp_activ_fn
        self.mp_residual = mp_residual
        self.mp_return_atten_weights = mp_return_atten_weights
        self.mp_edge_feat_atten = mp_edge_feat_atten

        ## config for interact module ##
        self.num_interact_transf_atten = num_interact_transf_atten
        self.seq_max_length = seq_max_length
        self.num_chunks = num_chunks
        self.init_channels = init_channels
        self.num_channels = num_channels
        self.use_region_atten = use_region_atten
        self.region_atten_n_head = region_atten_n_head
        self.interact_activ_fn = interact_activ_fn
        self.interact_dropout = interact_dropout

        ## config for pred head
        self.num_aa_classes=num_aa_classes
        self.num_ss_classes=num_ss_classes
        self.num_rsa_classes=num_rsa_classes
        self.num_dist_classes=num_dist_classes
        self.aa_loss_weight = aa_loss_weight
        self.ss_loss_weight = ss_loss_weight
        self.rsa_loss_weight = rsa_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.use_class_weights = use_class_weights

    def to_dict(self):
        output = super().to_dict()
        return output

@registry.register_task_model('seq_structure_multi_task', 'lm_mp')
@registry.register_task_model('multitask_fitness_UNsupervise_mutagenesis', 'lm_mp')
class SeqStructureMultiTaskModel(BaseModel):
    """Self-supervised learning based on structure information

    Multiple prediction tasks: 
        amino acid identities,
        secondary structure identities,
        relative solvent accessibility identities,
        distance-map
    """
    # define config class for model
    config_class = StructureMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config) # output: sequence_output, pooled_output, (hidden_states), (attentions)
        # check bert module parameter names
        #bert_param_names = self.bert.state_dict().keys()  

        ## freeze bert layers ##
        self.freeze_bert_partial(config.freeze_bert_firstN)

        ## graph module ##
        if config.mp_layer_name is not None:
            if config.mp_edge_feat_atten in ['s2t','t2s']:
                _input_edge_size = config.num_hidden_layers*config.num_attention_heads+config.graph_edge_pos_size
            elif config.mp_edge_feat_atten in ['both']:
                _input_edge_size = config.num_hidden_layers*config.num_attention_heads*2+config.graph_edge_pos_size
            else:
                raise Exception(f'Unacceptable value for mp_edge_feat_atten: {config.mp_edge_feat_atten}')
            self.graph_module = graph_MPNN(
                                    input_node_size=config.hidden_size+config.graph_node_pos_size,
                                    input_edge_size=_input_edge_size,
                                    lm_hidden_size=config.hidden_size,
                                    node_feat_channels=config.node_feat_channels,
                                    edge_feat_channels=config.edge_feat_channels,
                                    atten_heads=config.mp_atten_heads,
                                    dropout_rate=config.mp_dropout_rate,
                                    atten_dropout_rate=config.mp_atten_dropout_rate,
                                    add_self_loop=config.add_self_loop,
                                    mp_layer_name=config.mp_layer_name,
                                    num_layers=config.mp_num_layers,
                                    activ_fn=config.mp_activ_fn,
                                    residual=config.mp_residual,
                                    return_atten_weights=config.mp_return_atten_weights)
        else:
            self.graph_module = None
        ## interaction module ##
        self.interact_reduct = nn.Conv2d(config.hidden_size*2, config.init_channels, kernel_size=(1, 1), padding=(0, 0))
        self.dilated_resnet_1d = ResNet1DInputWithOptAttention(
                                    num_chunks=config.num_chunks,
                                    init_channels=config.init_channels,
                                    num_channels=config.num_channels,
                                    num_interact_transf_atten = config.num_interact_transf_atten,
                                    seq_max_length = config.seq_max_length,
                                    use_region_atten=config.use_region_atten,
                                    n_head=config.region_atten_n_head,
                                    activ_fn=config.interact_activ_fn,
                                    dropout=config.interact_dropout,
                                    initializer_range=config.initializer_range)
        self.dilated_resnet_2d = ResNet2DInputWithOptAttention(
                                    num_chunks=config.num_chunks,
                                    init_channels=config.init_channels,
                                    num_channels=config.num_channels,
                                    use_region_atten=config.use_region_atten,
                                    n_head=config.region_atten_n_head,
                                    activ_fn=config.interact_activ_fn,
                                    dropout=config.interact_dropout)

        ## prediction head ##
        self.pred_head = structureMultiTask_head(
                            num_chunks=1,
                            num_channels=config.num_channels,
                            num_aa_classes=config.num_aa_classes,
                            num_ss_classes=config.num_ss_classes,
                            num_rsa_classes=config.num_rsa_classes,
                            num_dist_classes=config.num_dist_classes,
                            use_region_atten=config.use_region_atten,
                            n_head_region_atten=config.region_atten_n_head,
                            activ_fn=config.interact_activ_fn,
                            dropout=config.interact_dropout,
                            ignore_index=-1,
                            aa_loss_weight=config.aa_loss_weight,
                            ss_loss_weight=config.ss_loss_weight,
                            rsa_loss_weight=config.rsa_loss_weight,
                            dist_loss_weight=config.dist_loss_weight,
                            use_class_weights=config.use_class_weights)

    def _resize_token_embeddings(self, new_num_tokens):
        """Resize token embedding dictionary
        copied from class BertModel
        """
        old_embeddings = self.bert.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.bert.embeddings.word_embeddings = new_embeddings
        return self.bert.embeddings.word_embeddings

    def _init_weights(self, module):
        """Initialize the weights
        copied from class BertAbstractModel 
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_bert_partial(self, freeze_bert_firstN):
        # freeze embedding layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # freeze pooler layers (MLP for first token's embedding)
        for param in self.bert.pooler.parameters():
            param.requires_grad = False

        # freeze first N attention layers in bert
        for i in range(freeze_bert_firstN):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        return NotImplemented
    
    def construct_interaction_tensor(self, node_feat: torch.Tensor):
        """Construct intra protein pairwise feature tensor

        Args:
            node_feat: feature tensor generated by bert or graph model, size: [batch_size,seq_length_padded,hidden_dim]

        Outputs:
            pairwise feature tensor of size [batch_size,2*hidden_dim,seq_length_padded,seq_length_padded]
        """
        new_node_feat = node_feat.permute(0,2,1)
        seq_pad_len = node_feat.shape[1]
        # interact_tensor size [batch_size,2*hidden_dim,seq_length,seq_length]
        interact_tensor = torch.cat((torch.repeat_interleave(new_node_feat.unsqueeze(3), repeats=seq_pad_len, dim=3),
                                    torch.repeat_interleave(new_node_feat.unsqueeze(2), repeats=seq_pad_len, dim=2)), dim=1)
        interact_tensor_t = interact_tensor.transpose(2,3) # sharing its underlying storage with the input tensor (since input is strided tensor)
        triu_idx_i,triu_idx_j =  torch.triu_indices(seq_pad_len, seq_pad_len, 1)
        interact_tensor_t[:,:,triu_idx_i,triu_idx_j] = interact_tensor[:,:,triu_idx_i,triu_idx_j] # interact_tensor is also changed
        interact_tensor.to(node_feat.device)
        #assert (interact_tensor.transpose(2,3) == interact_tensor).all() == True
        return interact_tensor
       
        
    def forward(self,
                input_seq_ids,
                input_seq_mask,
                aa_seq_mask,
                graph_batch, #pyg Batch object,
                graph_batch_idxs=None,
                token_type_ids=None,
                targets_seq=None,
                targets_ss=None,
                targets_rsa=None,
                targets_dist=None):
        bert_outputs = self.bert(input_seq_ids, input_mask=input_seq_mask, token_type_ids=token_type_ids)
        # bert_last_output:[batch_size,seq_length_padded,hidden_dim]; 
        # bert_attentions: (layer_1_attention, layer_2_attention,...)
        # each layer_i_attention: [batch_size,num_attention_head,seq_length,seq_length]
        residue_embeddings, bert_attentions = bert_outputs[0], bert_outputs[-1]
        outputs = (residue_embeddings, bert_attentions,)
        if self.graph_module is not None:
            bert_attentions = torch.permute(torch.stack(bert_attentions,dim=0), (1,0,2,3,4)) #[batch_size,num_layer,num_attention_head,seq_length,seq_length]
            bert_hidden_dim = residue_embeddings.size(2)
            ## graph module ##
            # graph node feature = lm_embedding + pos_encoding
            # graph edge feature = attention_weights + relative_pos_encoding
            graph_data_list = []
            if graph_batch_idxs is None:
                graph_batch_idxs = torch.arange(graph_batch.num_graphs,dtype=torch.int16).to(input_seq_ids.device)
            for graph_i, batch_i in zip(graph_batch_idxs,range(input_seq_ids.size(0))):
                graph_data = graph_batch.get_example(graph_i).to(input_seq_ids.device)
                node_pos_encode = graph_data.pos # [n_i,1]
                edge_pos_encode = graph_data.edge_pos # [e_i,1]
                graph_edge_index = graph_data.edge_index # [2,e_i]
                # extract amino acid embeddings from padded sequence
                aa_seq_mask_extend = aa_seq_mask[batch_i].unsqueeze(-1).expand(-1,bert_hidden_dim).bool()
                aa_seq_embeddings = torch.masked_select(residue_embeddings[batch_i],aa_seq_mask_extend).reshape(-1,bert_hidden_dim) # [l_i,hidden_dim]
                graph_data.x = torch.cat((aa_seq_embeddings,node_pos_encode),dim=1)
                # extract attention weights for edge
                # s2t: source's attention on target; t2s: target's attention on source
                graph_edge_attr_s2t = torch.stack([bert_attentions[batch_i,:,:,graph_edge_index[0,e].item(),graph_edge_index[1,e].item()].reshape(-1) for e in range(graph_edge_index.size(1))],dim=0) # [e_i,num_atten_weights]
                graph_edge_attr_t2s = torch.stack([bert_attentions[batch_i,:,:,graph_edge_index[1,e].item(),graph_edge_index[0,e].item()].reshape(-1) for e in range(graph_edge_index.size(1))],dim=0) # [e_i,num_atten_weights]
                if self.config.mp_edge_feat_atten == 't2s':
                    graph_data.edge_attr = torch.cat((graph_edge_attr_t2s,edge_pos_encode),dim=1)
                elif self.config.mp_edge_feat_atten == 's2t':
                    graph_data.edge_attr = torch.cat((graph_edge_attr_s2t,edge_pos_encode),dim=1)
                elif self.config.mp_edge_feat_atten == 'both':
                    graph_data.edge_attr = torch.cat((graph_edge_attr_s2t,graph_edge_attr_t2s,edge_pos_encode),dim=1)
                graph_data_list.append(graph_data)
            ## pyg batching without padding the graph to same number of nodes
            graph_batch = pyg_Batch.from_data_list(graph_data_list)
            graph_attention_tuple = None
            if self.graph_module.return_atten_weights:
                graph_batch, graph_attention_tuple = self.graph_module(graph_batch)
            else:
                graph_batch = self.graph_module(graph_batch)
            outputs = outputs + (graph_batch, graph_attention_tuple,) 
            ## prepare padded embeddings along sequence dimension for interaction module
            residue_embeddings_tuple = ()
            for batch_i in range(input_seq_ids.size(0)):
                graph_data = graph_batch.get_example(batch_i).to(input_seq_ids.device)
                aa_seq_embeddings = graph_data.x # [num_nodes,graph_h_d]
                aa_seq_mask_extend = aa_seq_mask[batch_i].unsqueeze(-1).expand(-1,bert_hidden_dim).bool()
                special_token_embeddings = torch.masked_select(residue_embeddings[batch_i],~aa_seq_mask_extend).reshape(-1,bert_hidden_dim)
                residue_embeddings = torch.cat((special_token_embeddings[0,:].reshape(-1,bert_hidden_dim),aa_seq_embeddings.reshape(-1,bert_hidden_dim),special_token_embeddings[1:,:].reshape(-1,bert_hidden_dim)),dim=0)
                residue_embeddings_tuple += (residue_embeddings.unsqueeze(0),)
            residue_embeddings = torch.cat(residue_embeddings_tuple,dim=0)
        ## interaction module and task head ##  
        interact_tensor = self.construct_interaction_tensor(residue_embeddings)
        interact_tensor_reduct = self.interact_reduct(interact_tensor)
        dilat_res2d_output = self.dilated_resnet_2d(interact_tensor_reduct)
        dilat_res1d_output = self.dilated_resnet_1d(interact_tensor_reduct,aa_seq_mask)
        outputs = outputs + (dilat_res1d_output, dilat_res2d_output,)
        output_pred = self.pred_head(dilat_res1d_output,dilat_res2d_output,targets_seq,targets_ss,targets_rsa,targets_dist)
        # outputs: ((loss, metric), (aa, ss, rsa, dist logits), bert_embeddings, bert_attentions, graph_output, graph_attentions,dilated_res1d_embeddings, dilated_res2d_embeddings)
        outputs = output_pred + outputs
        return outputs
