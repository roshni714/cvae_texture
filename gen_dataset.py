from spriteworld import factor_distributions as distribs
from spriteworld import renderers as spriteworld_renderers
from spriteworld import sprite_generators
from PIL import Image

def gen_sprite_dataset(mode="train"):
  common_factors = distribs.Product([
      distribs.Continuous('x', 0.1, 0.9),
      distribs.Continuous('y', 0.1, 0.9),
      distribs.Continuous('angle', 0, 360, dtype='int32'),
  ])

  # train/test split for goal-finding object scales and clustering object colors
  goal_finding_scale_test = distribs.Continuous('scale', 0.08, 0.12)
  green_blue_colors = distribs.Product([
      distribs.Continuous('c1', 64, 256, dtype='int32'),
      distribs.Continuous('c2', 64, 256, dtype='int32'),
  ])
  if mode == 'train':
    goal_finding_scale = distribs.SetMinus(
        distribs.Continuous('scale', 0.05, 0.15),
        goal_finding_scale_test,
    )
    cluster_colors = distribs.Product(
        [distribs.Continuous('c0', 128, 256, dtype='int32'), green_blue_colors])
  elif mode == 'test':
    goal_finding_scale = goal_finding_scale_test
    cluster_colors = distribs.Product(
        [distribs.Continuous('c0', 0, 128, dtype='int32'), green_blue_colors])
  else:
    raise ValueError(
        'Invalid mode {}. Mode must be "train" or "test".'.format(mode))

  # Create clustering sprite generators
  sprite_gen_list = []
  cluster_shapes = [
      distribs.Discrete('shape', [s])
      for s in ['triangle']
  ]
  for shape in cluster_shapes:
    factors = distribs.Product([
        common_factors,
        cluster_colors,
        shape,
        distribs.Continuous('scale', 0.08, 0.12),
    ])
    sprite_gen_list.append(
        sprite_generators.generate_sprites(factors, num_sprites=1))

  # Concat clusters into single scene to generate
  sprite_gen = sprite_generators.chain_generators(*sprite_gen_list)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)

  drawer = spriteworld_renderers.PILRenderer(
              image_size=(64, 64), anti_aliasing=5)

  print("Generating Sprites Dataset")
  for i in range(2000):
     sprites = sprite_gen()
     arr = drawer.render(sprites=sprites)
     img = Image.fromarray(arr, 'RGB')
     img.save('datasets/sprites1_val/img_{}.png'.format(i)) 
gen_sprite_dataset()
