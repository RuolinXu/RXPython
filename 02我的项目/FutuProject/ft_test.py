import futuquant.futuquant.open_context as oc
import futuquant.futuquant.quote_query


if __name__ == '__main__':
    quote_ctx = oc.OpenQuoteContext(host='127.0.0.1', port=11111)
    # ret_code, data = qt.get_market_snapshot(['US.NVDA'])
    # data.to_csv("index_sh.txt", index=True, sep=' ', columns=data.columns)
    # print(data)
    # ret_code, ret_data = quote_ctx.get_plate_list('HK', 'CONCEPT')
    # ret_code, ret_data = quote_ctx.get_plate_stock('HK.BK1110')

    ret_code, ret_data = quote_ctx.get_history_kline('US.BABA', start='1990-01-01', end='2100-01-01', ktype='K_DAY',
                                                     autype='qfq')
    print(ret_data)
    quote_ctx.close()


