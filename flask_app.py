import io
import db_util as db
import pandas as pd
import mysql.connector
# from global_ml import check
import ml_workflow as mlw
import upside_clustering as uc
from flask_session import Session
from flask import Flask, render_template, session, request

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

@app.route('/validate_logs', methods=['POST'])
def validate_logs_page():
    
    acc_dict = session.get('acc_dict', {})
    gyro_dict = session.get('gyro_dict', {})
    train = session.get('train_log', {})
    train_key_list = [k for k in train] # list of log_id used for the training

    logs = session.get('logs', {}) # all log (train + test)

    # retrive the logs that are not used for training 
    # use them for validation
    acc_logs = {}
    gyro_logs = {}
    for sensor_key in logs.keys():
        if sensor_key == 'Accelerometer':
            acc_logs[sensor_key] = [log_id for log_id in logs[sensor_key] if log_id not in train_key_list]
        else:
            gyro_logs[sensor_key] = [log_id for log_id in logs[sensor_key] if log_id not in train_key_list]
    
    print("\nvalidate_logs_page() - acc_logs")
    print(acc_logs)
    print("\nvalidate_logs_page() - gyro_logs")
    print(gyro_logs)


    # retrive all logs that are not used for training and validation
    # use them as general log for validation 
    # they relate to activities other than those on which the model was trained 
    general_acc_logs = {'Accelerometer': [k for k, v in acc_dict.items()]}
    general_gyro_logs = {'Gyroscope': [k for k, v in gyro_dict.items()]}
    logs = session.get('logs', {})
    none_activity_logs = [k for k, v in acc_dict.items() if v['activity'] == "None"]

    # remove the logs that are used for training and validation
    # and those that are not related to any activity
    for sensor_key, log_ids in logs.items():
        if sensor_key in general_acc_logs:
            general_acc_logs[sensor_key] = [log_id for log_id in general_acc_logs[sensor_key] if (log_id not in log_ids and log_id not in none_activity_logs)]
        if sensor_key in general_gyro_logs:
            general_gyro_logs[sensor_key] = [log_id for log_id in general_gyro_logs[sensor_key] if (log_id not in log_ids and log_id not in none_activity_logs)]
    
    threshold = session['training_loss']
    event_name = session['event_name']
    acc_eval_res, acc_conf_matrix = mlw.eval_models("Accelerometer", threshold, acc_logs, general_acc_logs, session['chosen_model'], session['model_name'], session['scaler'], session['event_name'])
    gyro_eval_res, gyro_conf_matrix = mlw.eval_models("Gyroscope", threshold, gyro_logs, general_gyro_logs, session['chosen_model'], session['model_name'], session['scaler'], session['event_name'])

    session['acc_conf_matrix'] = acc_conf_matrix
    session['gyro_conf_matrix'] = gyro_conf_matrix 

    return render_template('train_and_validation.html', train_sensor = session['sensor'], acc_dict = acc_dict, gyro_dict = gyro_dict, acc_eval_result = acc_eval_res, gyro_eval_result = gyro_eval_res, acc_conf_mat = session['acc_conf_matrix'], gyro_conf_mat = session['gyro_conf_matrix'], show_eval = True)

@app.route('/process_logs', methods=['POST'])
def process_logs(): 
    parameters = request.form
    epochs = int(parameters.get('epochs'))
    batch_size = int(parameters.get('batch_size'))
    percentage = int(parameters.get('percentage'))
    sensor = parameters.get('sensor')
    model = parameters.get('model')

    logs = session.get('logs', {})
    
    if sensor != 'Both':
        sub_logs = {sensor: logs.get(sensor, [])}
    else: 
        sub_logs = logs

    chosen_model, scaler, training_loss, test_log, train_log = mlw.model_train_activity(sub_logs, epochs, batch_size, percentage, model)
    session['model_name'] = model
    session['chosen_model'] = chosen_model
    session['scaler'] = scaler
    session['test_log'] = test_log
    session['train_log'] = train_log
    session['sensor'] = sensor
    session['training_loss'] = training_loss    


    return render_template('train_and_validation.html', logs = logs, chosen_model = chosen_model, scaler = scaler, test_log = test_log, selected_activity = session['selected_activity'])

@app.route('/process_activity', methods=['POST'])
def process_activity():
    selected_activity = request.form['activity_log']
    session['selected_activity'] = selected_activity
    
    acc_dict = session.get('acc_dict', {})
    gyro_dict = session.get('gyro_dict', {})

    logs = {
    'Accelerometer': [k for k, v in acc_dict.items() if v['activity'] == selected_activity],
    'Gyroscope': [k for k, v in gyro_dict.items() if v['activity'] == selected_activity]
    }

    session['logs'] = logs

    if 'acc_pca_plt' in session and 'gyro_pca_plt' in session:  
        return render_template('train_and_validation.html', selected_activity = selected_activity, logs = logs, results= session['result_users'], activities = session['activities'], user_movement = session['user_movement'], acc_pca = session['acc_dict'], gyro_pca = session['gyro_dict'], plt_acc_pca = session['acc_pca_plt'], plt_gyro_pca = session['gyro_pca_plt'])
    elif 'acc_plt' in session and 'gyro_plt' in session: 
        return render_template('train_and_validation.html', selected_activity = selected_activity, logs = logs, results= session['result_users'], activities = session['activities'], user_movement = session['user_movement'], acc_dict = session['acc_dict'], gyro_dict = session['gyro_dict'], plt_acc = session['acc_plt'], plt_gyro = session['gyro_plt'])

@app.route('/variance', methods=['POST'])
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

    parameters = request.form
    acc_iteration = int(parameters.get('acc_iteration'))
    gyro_iteration = int(parameters.get('gyro_iteration'))

    acc_df_0, acc_df_1, plt_acc = uc.ml_kmeans(df_accelerometer, "Accelerometer", acc_iteration)
    gyro_df_0, gyro_df_1, plt_gyro = uc.ml_kmeans(df_gyroscope, "Gyroscope", gyro_iteration)

    # Save ID_movement and corresponding variances in a dictionary
    acc_dict = {}
    for _, row in acc_df_1.iterrows():
        sql = "select activity_name from activity where ID_activity = (select ID_activity from movement where ID_movement = %s)"
        cursor.execute(sql, (row['ID_movement'],))
        acc_act = cursor.fetchone()
        
        acc_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ'], 
            'activity': acc_act['activity_name'] if acc_act['activity_name'] else "None"
        }

    # Save ID_movement and corresponding variances in a dictionary
    gyro_dict = {}
    for _, row in gyro_df_1.iterrows():
        sql = "select activity_name from activity where ID_activity = (select ID_activity from movement where ID_movement = %s)"
        cursor.execute(sql, (row['ID_movement'],))
        gyro_act = cursor.fetchone()
        gyro_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ'], 
            'activity': gyro_act['activity_name'] if gyro_act['activity_name'] else "None"
        }


    cursor.close()
    connection.close()

    session['acc_dict'] = acc_dict
    session['gyro_dict'] = gyro_dict
    session['acc_plt'] = plt_acc
    session['gyro_plt'] = plt_gyro

    return render_template('train_and_validation.html', results= session['result_users'], activities = session['activities'], user_movement = session['user_movement'], acc_dict = acc_dict, gyro_dict = gyro_dict, plt_acc = plt_acc, plt_gyro = plt_gyro)

@app.route('/variance_pca', methods = ['POST'])
def variance_pca_page(): 
    print("Variance PCA page")
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

    parameters = request.form
    acc_iteration = int(parameters.get('acc_iteration'))
    gyro_iteration = int(parameters.get('gyro_iteration'))

    acc_df_0, acc_df_1, plt_acc_pca = uc.PCA_kmeans(df_accelerometer, "Accelerometer", acc_iteration)
    gyro_df_0, gyro_df_1, plt_gyro_pca = uc.PCA_kmeans(df_gyroscope, "Gyroscope", gyro_iteration)

    # Save ID_movement and corresponding variances in a dictionary
    acc_dict = {}
    for _, row in acc_df_1.iterrows():
        sql = "select activity_name from activity where ID_activity = (select ID_activity from movement where ID_movement = %s)"
        cursor.execute(sql, (row['ID_movement'],))
        acc_act = cursor.fetchone()
        
        acc_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ'], 
            'activity': acc_act['activity_name'] if acc_act['activity_name'] else "None"
        }

    # Save ID_movement and corresponding variances in a dictionary
    gyro_dict = {}
    for _, row in gyro_df_1.iterrows():
        sql = "select activity_name from activity where ID_activity = (select ID_activity from movement where ID_movement = %s)"
        cursor.execute(sql, (row['ID_movement'],))
        gyro_act = cursor.fetchone()

        gyro_dict[row['ID_movement']] = {
            'varianceX': row['varianceX'],
            'varianceY': row['varianceY'],
            'varianceZ': row['varianceZ'], 
            'activity': gyro_act['activity_name'] if gyro_act['activity_name'] else "None"
        }

    cursor.close()
    connection.close()

    session['acc_dict'] = acc_dict
    session['gyro_dict'] = gyro_dict
    session['acc_pca_plt'] = plt_acc_pca
    session['gyro_pca_plt'] = plt_gyro_pca

    return render_template('train_and_validation.html', results = session['result_users'], activities = session['activities'], events = session['event_name'], user_movement = session['user_movement'], acc_pca = acc_dict, gyro_pca = gyro_dict, plt_acc_pca = plt_acc_pca, plt_gyro_pca = plt_gyro_pca)

@app.route('/train_and_validation', methods = ['POST'])
def train_and_validation_page():
    session.clear()
    event_name = request.form['button']
    session['event_name'] = event_name 
    
    event_id = db.get_eventID_by_event_name(event_name)
    session['event_id'] = event_id

    results = db.get_userIDs_by_eventID(event_id)
    session['result_users'] = results
    print(results)

    user_movement_list = db.get_userIDs_and_movementID_by_eventID(event_id)
    session['user_movement'] = user_movement_list

    activities = db.get_activityName_by_eventID(event_id)
    session['activities'] = activities

    if event_name == "UPSIDE":
        return render_template('train_and_validation.html', activities = ['activities'], event_name = session['event_name'], results= session['result_users'], user_movement = session['user_movement'])
    else: 
        devices = db.get_devices_by_eventID(event_id)
        session['devices'] = devices

        return render_template('train_and_validation_meta.html', event_name = session['event_name'], results = session['result_users'], activities = session['activities'], devices = session['devices'], user_movement = session['user_movement'])

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
    session['model_name'] = model

    model, scaler, training_loss = mlw.model_train(userid, sid, device, features, epochs, batch_size, model)
    session['user_id'] = userid
    session['model'] = model
    session['scaler'] = scaler
    session['device'] = device
    session['features'] = features
    session['training_loss'] = training_loss

    result = session.get('result_users', [])
    device = session.get('devices', [])
    event = session.get('events', [])
    return render_template('train_and_validation_meta.html', results = result, devices = device, events = event, user_movement = session['user_movement'])

@app.route('/validate_mse', methods=['POST'])
def validate():
    parameters = request.form
    eval_sid = list(parameters.get('eval_sid').split(','))
    model = session.get('model')
    scaler = session.get('scaler')
    device = session.get('device')
    features = session.get('features', [])
    model_name = session.get('model_name')
    threshold = session.get('training_loss')
    userid = session.get('user_id')

    mses, df_dict, path = mlw.meta_eval_models(model, threshold, userid, scaler, device, features, eval_sid, session['event_name'], model_name)

    result = session.get('result_users', [])
    device = session.get('devices', [])

    if features in ['Pitch', 'Roll', 'Yaw']:
        sensor = 'Orientation'
    else: 
        sensor = 'Position'

    if isinstance(mses, pd.DataFrame) and not mses.empty:
        return render_template('train_and_validation_meta.html', train_sensor = sensor, eval_result = df_dict, meta_conf_mat = path, activities = session['activities'], results=result, user_movement = session['user_movement'], devices=device, events=session['event_name'])
    elif mses:
        return render_template('train_and_validation_meta.html', train_sensor = sensor, eval_result = df_dict, meta_conf_mat = path, activities = session['activities'], results=result, user_movement = session['user_movement'], devices=device, events=session['event_name'])
    else:
        return render_template('train_and_validation_meta.html', results=result, user_movement = session['user_movement'], devices=device, activities = session['activities'], events=session['event_name'])

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