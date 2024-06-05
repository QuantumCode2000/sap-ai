from roboflow import Roboflow
import supervision as sv
import cv2
from dotenv import load_dotenv
import os
load_dotenv()
tensorflow_api_key = os.getenv("TENSORFLOW_API_KEY")
modelo_mejor_acuracy = os.getenv("MODELO_MEJOR_ACURACY")
# print(tensorflow_api_key)
rf = Roboflow(api_key=tensorflow_api_key)
project = rf.workspace().project("persona-bajo-la-lluvia")
model = project.version(1).model
#



#le damos la imagen para que haga la prediccion de las etiquetas

result = model.predict("./data/ag_sex1.png", confidence=40, overlap=30).json()
#
labels = [item["class"] for item in result["predictions"]]

print("caracteristicas que detecta el modelo:")
print(labels)
# print(result)





# para poder ver resaltado en la imagen lo que detecta el modelo
detections = sv.Detections.from_roboflow(roboflow_result=result)


# Crear instancias de los anotadores
# label_annotator = sv.LabelAnnotator()  # Crear un anotador de etiquetas (Versión original)
la = sv.LabelAnnotator()  # Crear un anotador de etiquetas
bba = sv.BoxAnnotator()  # Crear un anotador de cajas delimitadoras

# Leer la imagen desde el archivo "./data/ag_sex1.png"
image = cv2.imread("./data/ag_sex1.png")


# Anotar la imagen con cajas delimitadoras resaltadas
annotated_image = bba.annotate(
    scene=image, detections=detections)

# Anotar la imagen con etiquetas resaltadas
annotated_image = la.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# Mostrar la imagen anotada
sv.plot_image(image=annotated_image, size=(16, 16))

print(
ModeloCNN2.evaluate(X_test, y_test))