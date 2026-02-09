use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct DeepFeedForward<B: Backend> {
    // We name them explicitly to simulate distinct memory blocks
    pub layer_01: Linear<B>,
    pub layer_02: Linear<B>,
    pub layer_03: Linear<B>,
    pub layer_04: Linear<B>,
    pub layer_05: Linear<B>,
    pub layer_06: Linear<B>,
    pub layer_07: Linear<B>,
    pub layer_08: Linear<B>,
    pub layer_09: Linear<B>,
    pub layer_10: Linear<B>,
    pub output_layer: Linear<B>,
}

// Configuration struct (Hyperparameters)
#[derive(burn::config::Config)]
pub struct DeepFeedForwardConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl DeepFeedForwardConfig {
    // This "init" function allocates the memory for the weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeepFeedForward<B> {
        DeepFeedForward {
            layer_01: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            layer_02: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_03: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_04: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_05: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_06: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_07: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_08: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_09: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            layer_10: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            output_layer: LinearConfig::new(self.hidden_size, self.output_size).init(device),
        }
    }
}

impl<B: Backend> DeepFeedForward<B> {
    // The Standard Forward Pass (All layers in memory at once)
    // Later, we will REPLACE this with a "Streaming Forward Pass"
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.layer_01.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_02.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_03.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_04.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_05.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_06.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_07.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_08.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_09.forward(x); x = burn::tensor::activation::relu(x);
        x = self.layer_10.forward(x); x = burn::tensor::activation::relu(x);

        self.output_layer.forward(x)
    }
}