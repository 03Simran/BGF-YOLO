from roboflow import Roboflow
rf = Roboflow(api_key="p5L4XnvSTpIxVVZcbJsx")
project = rf.workspace("aadi-5hpnv").project("plant-diseases-fbvex")
version = project.version(2)
dataset = version.download("yolov8")