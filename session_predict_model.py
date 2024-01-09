import pickle
from cat_auxiliary_functions import make_pred
from flask import Flask, render_template, request

app = Flask(__name__)

# load model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)
    
with open('model_w_difficulty.pkl', 'rb') as file:
    model_w_difficulty = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    difficulty_level = None
    if request.method == 'POST':
        # get inputs
        selected_value = request.form['dropdown']
        difficulty_level = int(request.form['dropdown2'])
                
        # make prediction
        result = make_pred(selected_value, model_w_difficulty, difficulty_level)[0]
        
        # convert binary input to string
        difficulty_level = "harder" if difficulty_level else "same"
        
        return render_template('index.html', result=result, selected_value=selected_value, selected_difficulty=difficulty_level)
    return render_template('index.html', result=None, selected_value=None, selected_difficulty=difficulty_level)

if __name__ == '__main__':
    app.run(debug=True)
