import fire

from dc3.model.full_pipeline import DC3Pipeline

def train(overwrite=False):
  pipeline = DC3Pipeline(overwrite=overwrite)
  if not pipeline.is_trained:
    pipeline.fit_end2end()

def inference():
  pass

if __name__=='__main__':
  fire.Fire()
