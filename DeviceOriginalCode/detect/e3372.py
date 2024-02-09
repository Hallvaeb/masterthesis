import requests
import xmltodict
import json
DEFAULT_HOSTNAME = 'http://192.168.8.1/api'
SIGNAL_ENDPOINT = '/device/signal'
DEVICE_INFORMATION_ENDPOINT = '/device/information'
TOKEN_ENDPOINT = '/webserver/SesTokInfo'
MONITORING_ENDPOINT = '/monitoring/status'


def get_device_information() -> str:
  """Returns all device properties """
  data = __retrieve_device_information()
  return json.dumps(data['response'])


def get_iccid() -> str:
  """Returns ICCID (Integrated Circuit Card Identifier)"""
  data = __retrieve_device_information()
  return data['response']['Iccid']


def get_signal() -> str:
  """Returns all signal properties"""
  data = __retrieve_device_signal()
  return json.dumps(data['response'])


def get_signal_values() -> str:
  """Returns all signal properties"""
  data = __retrieve_device_signal()
  loaded_data = data['response']
  rsrq = loaded_data['rsrq']
  rsrp = loaded_data['rsrp']
  rssi = loaded_data['rssi']
  sinr = loaded_data['sinr']
  mode = loaded_data['mode']
  cell_id = loaded_data['cell_id']
  return json.dumps({'rsrq': rsrq, 'rsrp': rsrp, 'rssi': rssi, 'sinr': sinr, 'mode': mode, 'cell_id': cell_id})


def get_imei() -> str:
  """Returns IMEI Number"""
  data = __retrieve_device_information()
  return data['response']['Imei']


def get_celltower_id() -> str:
  """Returns Cell Tower ID"""
  data = __retrieve_device_signal()
  return data['response']['cell_id']


def get_signal_level() -> str:
  data = __retrieve_monitoring_status()
  return data['response']['SignalIcon']


def get_signal_mode() -> str:
  """Returns signal mode (3G, 4G, LTE etc.)"""
  data = __retrieve_device_signal()
  mode = json.loads(data)['response']['mode']
  return __prettify_signal_mode(mode)


def __token():
  r = requests.get(DEFAULT_HOSTNAME + TOKEN_ENDPOINT)
  d = xmltodict.parse(r.text, xml_attribs=True)
  return d['response']


def __prettify_signal_mode(signal_mode: int) -> str:
  result = 'n/a'
  if signal_mode == '0':
    result = 'No Service'
  elif signal_mode == '1':
    result = 'GSM'
  elif signal_mode == '10':
    result = 'EV-DO rev. 0'
  elif signal_mode == '11':
    result = 'EV-DO rev. A'
  elif signal_mode == '12':
    result = 'EV-DO rev. B'
  elif signal_mode == '13':
    result = '1xRTT'
  elif signal_mode == '14':
    result = 'UMB'
  elif signal_mode == '15':
    result = '1xEVDV'
  elif signal_mode == '16':
    result = '3xRTT'
  elif signal_mode == '17':
    result = 'HSPA+ 64QAM'
  elif signal_mode == '18':
    result = 'HSPA+ MIMO'
  elif signal_mode in ['19', '101']:
    result = 'LTE (4G)'
  elif signal_mode == '2':
    result = 'GPRS (2.5G)'
  elif signal_mode == '3':
    result = 'EDGE (2.75G)'
  elif signal_mode == '4':
    result = 'WCDMA (3G)'
  elif signal_mode == '41':
    result = 'UMTS (3G)'
  elif signal_mode in ['45', '65']:
    result = 'HSPA+ (3G)'
  elif signal_mode == '46':
    result = 'DC-HSPA+ (3G)'
  elif signal_mode == '5':
    result = 'HSDPA (3G)'
  elif signal_mode == '6':
    result = 'HSUPA (3G)'
  elif signal_mode in ['7', '44', '64']:
    result = 'HSPA (3G)'
  elif signal_mode == '8':
    result = 'TD-SCDMA (3G)'
  elif signal_mode == '9':
    result = 'HSPA+ (4G)'
  return result


def __retrieve_device_information() -> str:
  """Fetches device information from modem API"""
  token = __token()
  headers = {'Cookie': token['SesInfo'], '__RequestVerificationToken': token['TokInfo']}
  request = requests.get(DEFAULT_HOSTNAME + DEVICE_INFORMATION_ENDPOINT, headers=headers)
  return __convert_xml_to_json(request.text)


def __retrieve_monitoring_status() -> str:
  token = __token()
  headers = {'Cookie': token['SesInfo'], '__RequestVerificationToken': token['TokInfo']}
  request = requests.get(DEFAULT_HOSTNAME + MONITORING_ENDPOINT, headers=headers)
  return __convert_xml_to_json(request.text)


def __retrieve_device_signal() -> str:
  """Fetches signal information from modem API"""
  token = __token()
  headers = {'Cookie': token['SesInfo'], '__RequestVerificationToken': token['TokInfo']}
  request = requests.get(DEFAULT_HOSTNAME + SIGNAL_ENDPOINT, headers=headers)
  return __convert_xml_to_json(request.text)


def __convert_xml_to_json(xml) -> str:
  """XML to JSON conversion"""
  return xmltodict.parse(xml)
