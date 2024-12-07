from flask import Flask, render_template, session, request
from global_ml import check
from flask_session import Session
import ml_workflow as mlw
import mysql.connector
import pandas as pd
import upside_clustering as uc

app = Flask(__name__)

app.config['SECRET_KEY'] = 'new_session_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

app.config['MYSQL_HOST'] = 'localhost'  
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = '' 
app.config['MYSQL_DATABASE'] = 'allDataDb'

def get_db_connection():
    connection = mysql.connector.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DATABASE']
    )
    return connection

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/variance')
def variance_page():
    print("Variance page")
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("SELECT ID_movement, varianceX, varianceY, varianceZ, sensor FROM movement where sensor = \"accelerometer\" and varianceX IS NOT NULL AND varianceY IS NOT NULL AND varianceZ IS NOT NULL")
    results = cursor.fetchall()
    df_accelerometer = pd.DataFrame(results)
    session['df_accelerometer'] = df_accelerometer

    cursor.execute("SELECT ID_movement, varianceX, varianceY, varianceZ, sensor FROM movement where sensor = \"gyroscope\" and varianceX IS NOT NULL AND varianceY IS NOT NULL AND varianceZ IS NOT NULL")
    results = cursor.fetchall()
    df_gyroscope = pd.DataFrame(results)
    session['df_gyroscope'] = df_gyroscope

    cursor.close()
    connection.close()

    acc_df_0, acc_df_1 = uc.ml_kmeans(df_accelerometer, "Accelerometer")
    gyro_df_0, gyro_df_1 = uc.ml_kmeans(df_gyroscope, "Gyroscope")

    # Save ID_movement and corresponding variances in a dictionary
    acc_dict = {}
    for _, row in acc_df_1.iterrows():
        acc_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ']
        }

    # Save ID_movement and corresponding variances in a dictionary
    gyro_dict = {}
    for _, row in gyro_df_1.iterrows():
        gyro_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ']
        }

    session['acc_dict'] = acc_dict
    session['gyro_dict'] = gyro_dict

    print(session['gyro_dict'])

    #print("Accelerometer effective movement log: ", acc_df_1)
    #print("Gyroscope effective movement log:", gyro_df_1)

    return render_template('train_and_validation.html', acc_dict = acc_dict, gyro_dict = gyro_dict)

@app.route('/train_and_validation')
def train_and_validation_page():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("SELECT ID FROM user where Name is NULL")  
    results = cursor.fetchall() 
    session['result_users'] = results

    cursor.execute("select distinct event_name from event where ID_event = 1")
    events = cursor.fetchall()
    session['events'] = events 

    cursor.execute(f"select activity_name from activity where ID_activity in (select a.ID_activity from association as a join event as e on a.ID_event and a.ID_event = 1)")
    activities = cursor.fetchall()
    session['activities'] = activities

    cursor.execute("select distinct type from device as d where d.ID_activity in (select distinct a.ID_activity from association as a join event as e on a.ID_event and a.ID_event = 1)")
    devices = cursor.fetchall()
    session['devices'] = devices
    
    sql = "select ID_user, ID_movement from movement where ID_user in (select ID from user where Name is NULL)"
    cursor.execute(sql)
    user_movement_list = cursor.fetchall()  
    # Convert list to dictionary
    user_movement = {}
    for row in user_movement_list:
        user_id = row['ID_user']
        movement_id = row['ID_movement']
        if user_id not in user_movement:
            user_movement[user_id] = []
        user_movement[user_id].append(movement_id)
    session['user_movement'] = user_movement_list
     
    cursor.close()
    connection.close()
    return render_template('train_and_validation.html', results= results, activities = activities , devices = devices, events = events, user_movement = user_movement)


@app.route('/run_train', methods=['POST'])
def run_train():
    parameters = request.form
    userid = int(parameters.get('userid'))
    sid = int(parameters.get('sid'))
    device = parameters.get('device')
    features = parameters.get('feature')
    epochs = int(parameters.get('epochs'))
    batch_size = int(parameters.get('batch_size'))
    model = parameters.get('model')

    model, scaler = mlw.model_train(userid, sid, device, features, epochs, batch_size, model)
    session['model'] = model
    session['scaler'] = scaler
    session['device'] = device
    session['features'] = features

    result = session.get('result_users', [])
    device = session.get('devices', [])
    event = session.get('events', [])
    return render_template('train_and_validation.html', results = result, devices = device, events = event)

@app.route('/validate_mse', methods=['POST'])
def validate():
    parameters = request.form
    eval_sid = list(parameters.get('eval_sid').split(','))
    model = session.get('model')
    scaler = session.get('scaler')
    device = session.get('device')
    features = session.get('features', [])
    print(eval_sid, device, features)
    mses = mlw.evaluate_model(model, scaler, device, features, eval_sid)
    session['result_list'] = mses
    result_list = session.get('result_list', {})
    result_list = result_list.groupby('user')['mse'].apply(list).to_dict()

    result = session.get('result_users', [])
    device = session.get('devices', [])
    event = session.get('events', [])

    if isinstance(result_list, pd.DataFrame) and not result_list.empty:
        return render_template('train_and_validation.html', result_list=result_list, results=result, devices=device, events=event)
    elif result_list:
        return render_template('train_and_validation.html', result_list=result_list, results=result, devices=device, events=event)
    else:
        return render_template('train_and_validation.html', result_list={}, results=result, devices=device, events=event)



@app.route('/check')
def check_page():
    return render_template('check.html')


@app.route('/check', methods=['POST'])
def run_check():
    parameters = request.form  # Se i dati sono inviati tramite il form
    userid = int(parameters.get('userid'))
    sid = int(parameters.get('sid'))
    parameter = parameters.get('parameter')
    threshold = float(parameters.get('threshold'))
    epochs = int(parameters.get('epochs'))
    batch_size = int(parameters.get('batch_size'))
    model = parameters.get('model')
    # Esegui il metodo check dallo script globale
    check(userid, sid, parameter, threshold, epochs, batch_size, model)
    return 'Check completed successfully'

@app.route('/run_experiments')
def run_experiments_page():
    return render_template('run_experiments.html')


@app.route('/run_experiments', methods=['POST'])
def run_experiments():
    parameters = request.form
    parameter = parameters.get('parameter')
    threshold = float(parameters.get('threshold'))
    epochs = int(parameters.get('epochs'))
    batch_size = int(parameters.get('batch_size'))
    model = parameters.get('model')
    # Esegui il metodo run_experiments dallo script globale
    run_experiments(parameter, threshold, epochs, batch_size,model)
    return 'Experiments completed successfully'

if __name__ == '__main__':
    app.run(debug=True)