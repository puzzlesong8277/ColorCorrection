from utils_diffusion import *


def main():

    diffusion_path = '../weights_final/DescanDiffusion.pth'
    color_encoder_path = 'exp/weights_color_encoder/color_encoder_6.pth'    
    test_folder = 'exp/test_ref'
    steps = 10

    model, color_encoder = load_diffusion_and_encoder(diffusion_path, color_encoder_path)
    beta = beta_schedule()
    alpha, alpha_hat = prepare_alpha_schedules(beta)
    
    test(test_folder, model, color_encoder, alpha, alpha_hat, steps=steps)

if __name__ == '__main__':
    main()