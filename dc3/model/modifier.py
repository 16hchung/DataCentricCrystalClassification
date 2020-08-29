from .full_pipeline import DC3Pipeline

dc3pipeline = DC3Pipeline()

def DC3Modifier(frame, data):
  lattice_labels = dc3pipeline.predict(data)
  data.particles_.create_property('_dc3label', data=lattice_labels)
