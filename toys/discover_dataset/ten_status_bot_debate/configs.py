import configparser

config = configparser.ConfigParser()
config.read('basic.conf')

people_lst = [
    'Sam',
    'George',
    'Lisa',
    'Wang',
    'Carol',
    'Tsukasa',
    'Hiru',
    'Meneoid',
    'Kate',
    'Bob'
]

stands = [
    '电商不应该存在，电商是对普通人生活的侵蚀',
    '电商是更先进的生产力，终将取代普通商业',
    'Sam的观点过于偏颇，专心反驳 Sam 的观点',
    '电商的发展目前是有偏的',
    '我们无法预知未来，我们不应该得出任何结论，我们应该继续观察'
]