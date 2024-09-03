import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    ## Device Agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## HYPERPARAMETERS
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 64
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    Z_DIM = 128
    NUM_EPOCHS = 5
    FEATURES_DISC = 64
    FEATURES_GEN = 64
    CRITIC_ITERATIONS = 5
    WEIGHT_CLIP = 0.01

    ## SETTING UP TRANSFORMS
    transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ])

    dataset = torchvision.datasets.ImageFolder(root='celeb_dataset/', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    critic = Wasserstein_distance(CHANNELS_IMG, FEATURES_DISC).to(device)

    initialize_weights(gen)
    initialize_weights(critic)

    ## SETTING UP OPTIMIZER
    optimizer_gen = torch.optim.RMSprop(params=gen.parameters(), lr=LEARNING_RATE)
    optimizer_critic = torch.optim.RMSprop(params=critic.parameters(), lr=LEARNING_RATE)

    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    writer_real = SummaryWriter(f"runs/real")
    writer_fake = SummaryWriter(f"runs/fake")

    save_path = './generated_samples'
    os.makedirs(save_path, exist_ok=True)

    total_batch = len(dataloader)

    step=0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_example, _) in enumerate(dataloader):
            
            gen.train()
            critic.train()

            real_example = real_example.to(device)
            
            for _ in range(CRITIC_ITERATIONS):
                
                ## TRAIN CRITIC: max E[c(real)] - E[c(generated)]
                
                noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
                fake_example = gen(noise)
                critic_value_on_real = critic(real_example).reshape(-1)
                critic_value_on_fake = critic(fake_example).reshape(-1)
                
                loss_critic = -(torch.mean(critic_value_on_real) - torch.mean(critic_value_on_fake))
                
                critic.zero_grad()
                
                loss_critic.backward(retain_graph=True)
                
                optimizer_critic.step()
                
                for p in critic.parameters():
                    p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
                
            ## TRAIN GENERATOR: min -E[c(generated)] <-> max E[c(generated)]
            
            output = critic(fake_example).reshape(-1)
            loss_gen = -torch.mean(output)
            
            gen.zero_grad()
            
            loss_gen.backward()
            
            optimizer_gen.step()
                
                
            if batch_idx%100 == 0:
                
                gen.eval()
                critic.eval()
                print(
                    f"Epoch[{epoch} / {NUM_EPOCHS}] | Batch[{batch_idx} / {total_batch}] \ "
                    f"Loss Critic: {loss_critic:.4f}, Generator Wasserstein Distance: {loss_gen:.4f}"
                )
                
                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_fake  = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real  = torchvision.utils.make_grid(real_example[:32], normalize=True)
                    
                    writer_fake.add_image(
                        "WGAN WC Fake Images", img_grid_fake, global_step=step
                    )
                    
                    writer_real.add_image(
                        "WGAN WC Real Images", img_grid_real, global_step=step
                    )
                    
                    save_file_path = os.path.join(save_path, f"ep{epoch}_bat{batch_idx}.png")
                    torchvision.utils.save_image(img_grid_fake, save_file_path)
                    
            step += 1

    model_path = 'model/generator_WGAN_WC.pth'
    torch.save(gen.state_dict(), model_path)

    model_path = 'model/critic_WGAN_WC.pth'
    torch.save(disc.state_dict(), model_path)
if __name__ == "__main__":
    main()