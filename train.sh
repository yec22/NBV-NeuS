### Blender Dataset
# train NeRF
# python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
# train NeuS with mask
# python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
# train NeuS without mask
# python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example system.loss.lambda_mask=0.0

### DTU Dataset
# train NeuS on DTU without mask
# python launch.py --config configs/neus-dtu.yaml --gpu 0 --train dataset.scene=dtu_scan105
# train NeuS on DTU with mask
python launch.py --config configs/neus-dtu-wmask.yaml --gpu 0 --train dataset.scene=dtu_scan105