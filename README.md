# mochi-video
 Etude on a Video Generative Model. 
 
 Open source code & architecture taken from https://www.genmo.ai/blog

# Model Architecture
Mochi 1 is a 10 billion parameter diffusion model built **Asymmetric Diffusion Transformer** architecture. The Mochi video VAE compresses videos to a 128x smaller size, with an 8x8 spatial and a 6x temporal compression to a 12-channel latent space.

An AsymmDiT takes text prompt and compressed video tokens, and jointly attends to text and visual tokens with multi-modal self-attention and learns separate MLP layers for each modality, similar to Stable Diffusion 3. 

The Mochi visual stream has nearly 4 times as many parameters as the text stream via a larger hidden dimension. To unify the modalities in self-attention, the researcher used **non-square QKV** and output projection layers. This asymmetric design reduces inference memory requirements.

Many modern diffusion models use multiple pretrained language models to represent user prompts. In contrast, Mochi 1 simply encodes prompts with a single T5-XXL language model.

Mochi 1 jointly reasons over a context window of 44,520 video tokens with full 3D attention. To localize each token, the researchers extend learnable rotary positional embeddings (RoPE) to 3-dimensions. 

The network end-to-end learns mixing frequencies for space and time axes.
Mochi benefits from some of the latest improvements in language model scaling including SwiGLU feedforward layers, query-key normalization for enhanced stability, and sandwich normalization to control internal activations.
