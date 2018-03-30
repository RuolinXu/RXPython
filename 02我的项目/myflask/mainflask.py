from flask import Flask, jsonify, request, render_template, url_for

app = Flask(__name__)


@app.route('/hello')
@app.route('/hello/<name>')
def hello_world(name=None):
    url1 = url_for('static', filename='code/highcharts.js')
    return render_template('test.html', name=name, url1=url1)
    # return 'Hello Flask!'


@app.route('/test', methods=['POST', 'GET'])
def new_transaction(sender=None):
    if request.method == 'POST':
        values = request.get_json()
        print(values)
        print("2222 %s" % sender)
        if values is None:
            response = {'message': f'Nonesssssss'}
            return jsonify(response), 201
        # Check that the required fields are in the POST'ed data
        required = ['sender', 'recipient', 'amount']
        if not all(k in values for k in required):
            response = {'message': f'Missing values'}
            return jsonify(response), 201

        # Create a new Transaction
        index = values['sender']

        response = {'message': f'Transaction will be added to Block {index}'}
        return jsonify(response), 201
    elif request.method == 'GET':
        return render_template('test.html')


@app.route('/')
def index():
    return render_template('index.html')


def chart2Data(df):
    df['AvgP'] = round(df.eval('Turnover / Volume'), 1)
    df['IsUp'] = df.eval('Close > Open')
    Udf, Ddf = df[df.Close > df.Open], df[df.Close < df.Open]
    up = Udf.groupby(['AvgP'])['Volume'].sum()
    down = Ddf.groupby(['AvgP'])['Volume'].sum()*(-1)
    up_categories, up_values = list(up.index.values), list(up.values)
    down_categories, down_values = list(down.index.values), list(down.values)
    return up_categories, up_values, down_categories, down_values


@app.route('/chart2',methods=['POST'])
def chart2():
    values = request.get_json()
    print(values)
    required = ['symbol', 'fromtime', 'totime']
    if values is None or not all(k in values for k in required):
        response = {'message': 'Missing values'}
        return jsonify(response), 501

    from XQ.StockData2 import StockData2
    pd = StockData2.kline_pd_from_db(values['symbol'], start=values['fromtime'], end=values['totime'], ktype='K_1M')
    up_categories, up_values, down_categories, down_values = chart2Data(pd)

    response = {'up_categories': up_categories, 'up_values': up_values,
                'down_categories': down_categories, 'down_values': down_values}
    return jsonify(response), 201


@app.route('/chart1',methods=['POST'])
def chart1():
    values = request.get_json()
    print(values)
    required = ['symbol', 'fromtime', 'totime']
    if values is None or not all(k in values for k in required):
        response = {'message': 'Missing values'}
        return jsonify(response), 501

    from XQ.StockData2 import StockData2
    pd = StockData2.kline_pd_from_db(values['symbol'], start=values['fromtime'], end=values['totime'], ktype='K_1M')
    i_length = len(pd)
    ranges = [[i, x[1]['Low'], x[1]['High']] for i, x in zip(range(i_length), pd.iterrows())]

    averages = [[i, x[1]['Close']] for i, x in zip(range(i_length), pd.iterrows())]
    response = {'ranges': ranges, 'averages': averages}
    return jsonify(response), 201


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)