import futuquant.futuquant.open_context as oc
import futuquant.futuquant.quote_query


if __name__ == '__main__':
    qt = oc.OpenQuoteContext(host='127.0.0.1', port=11111)
    ret_code, data = qt.get_market_snapshot(['US.NVDA180216C230000'])
    data.to_csv("index_sh.txt", index=True, sep=' ', columns=data.columns)
    print(data)

