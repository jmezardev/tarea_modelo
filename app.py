from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model.joblib')
#labels = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
    return """
    <html>

    <h1>Predicción de Préstamo </h1>
    <h3> Llene el siguiente formulario con los valores
    numéricos respectivos:</h3>

    <form action='predict' method='GET'>
    <label for="v1">Género:</label><br>
    <input type="text" id="v1" name="v1" placeholder="M(1)/F(0)"><br><br>
    <label for="v2">Está Casado?: </label><br>
    <input type="text" id="v2" name="v2" placeholder="No(0)/Si(1)"><br><br>
    <label for="v3">Cantidad de dependientes:</label><br>
    <input type="text" id="v3" name="v3"><br><br>
    <label for="v4">Educación: </label><br>
    <input type="text" id="v4" name="v4" placeholder="Grad.(0)/No grad.(1)"><br><br>

    <label for="v5">Trabajador independiente?: </label><br>
    <input type="text" id="v5" name="v5", placeholder="Si(1)/No(0)"><br><br>
    <label for="v6"> Ingreso del aplicante:</label><br>
    <input type="text" id="v6" name="v6"><br><br>
    <label for="v7">Ingreso del coaplicante:</label><br>
    <input type="text" id="v7" name="v7"><br><br>
    <label for="v8">Monto solicitado en miles:</label><br>
    <input type="text" id="v8" name="v8"><br><br>

    <label for="v9">Plazo del préstamo en meses</label><br>
    <input type="text" id="v9" name="v9"><br><br>
    <label for="v10">Tiene historial crediticio?: </label><br>
    <input type="text" id="v10" name="v10" placeholder="Si(1)/No(0)"><br><br>
    <label for="v11">Tipo de propiedad: </label><br>
    <input type="text" id="v11" name="v11" placeholder="Urb(2)/Semi(1)/Rur(0)"><br><br>



    <input type="submit">
    </form>

    </html>

    """

@app.route("/predict")
def predict():
    v1 = float(request.args.get('v1'))
    v2 = float(request.args.get('v2'))
    v3 = float(request.args.get('v3'))
    v4 = float(request.args.get('v4'))
    v5 = float(request.args.get('v5'))
    v6 = float(request.args.get('v6'))
    v7 = float(request.args.get('v7'))
    v8 = float(request.args.get('v8'))
    v9 = float(request.args.get('v9'))
    v10 = float(request.args.get('v10'))
    v11 = float(request.args.get('v11'))
    List=np.array([v1,v2,v3,v4, v5,v6,v7,v8,v9,v10,v11])
    List=List.reshape(-1, 1)

    result =int(model.predict(List.T))
    if result==0:
        return "<h1> Predicción: {}</h1>".format("No aplica al préstamo solicitado")
    else:
        return "<h1> Predicción: {}</h1>".format("Si aplica al préstamo solicitado")

if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
