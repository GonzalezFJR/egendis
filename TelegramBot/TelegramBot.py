'''
  Get event displays from a telegram chat bot !!!!
  This code is quite updated, but can be used as a skeleton to create a telegram app that creates event displays.
'''

import telepot
from Plot import GetEventDisplay, O
import time
from Help import textHelp

userinfo = {}

def handle(msg):
  chat_id = msg['chat']['id']
  command = msg['text'].lower()
  if len(command) > 10: return
  name = bot.getChat(chat_id)['first_name']
  if not chat_id in userinfo: userinfo[chat_id] = {}
  if not 'name' in userinfo[chat_id]: userinfo[chat_id]['name'] = name
  print ('[BOT] [%s] Got command: %s' %(name, command))
  command = command.lower()
  if command == '/start' or command == 'start':
    msg = 'Bienvenid@, %s, a esta clase de física de partículas de Con F de Física.'%name
    print(' >> ', msg)
    bot.sendMessage(chat_id, msg)
    bot.sendMessage(chat_id, textHelp)
  elif 'pu' in command:
    if   '=' in command: nPU = command.replace(' ', '').split('=')[-1]
    elif ':' in command: nPU = command.replace(' ', '').split(':')[-1]
    elif ' ' in command: nPU = command.split(' ')[-1]
    try:
      nPU = int(nPU)
    except:
      print('Invalid PU command... ', command)
      bot.sendMessage(chat_id, 'Comando no entendido!')
      nPU = O.nPU
    #O.nPU = nPU
    userinfo[chat_id]['nPU'] = nPU
    bot.sendMessage(chat_id, 'Número de trazas cambiado a %i'%nPU)
  elif command in ['ayuda', 'help']:
    bot.sendMessage(chat_id, textHelp)
  elif command == 'xxxinfo':
    for k in userinfo:
      print('{%s} '%str(k), userinfo[k])
  elif command.startswith('col'):
    #O.nPU = nPU
    nPU = int(O.nPU)
    if chat_id in userinfo and 'nPU' in userinfo[chat_id]:
      O.nPU = userinfo[chat_id]['nPU']
    outname = 'event_%s.png'%str(chat_id)
    report = GetEventDisplay('', outname, nPU=nPU)
    if O.status != '':
      bot.sendMessage(chat_id, O.status)
    else:
      bot.sendPhoto(chat_id, open(outname,'rb'))
    userinfo[chat_id]['report'] = report
  elif command.startswith('sol') or command.startswith('res') or command in ['?', '¿']:
    if 'report' in userinfo[chat_id]: bot.sendMessage(chat_id, userinfo[chat_id]['report'])
  else:
    nPU = int(O.nPU)
    if chat_id in userinfo and 'nPU' in userinfo[chat_id]:
      O.nPU = userinfo[chat_id]['nPU']
    outname = 'event_%s.png'%str(chat_id)
    report = GetEventDisplay(command, outname, nPU=nPU)
    userinfo[chat_id]['report'] = report
    time.sleep(0.4)
    if O.status != '':
      bot.sendMessage(chat_id, O.status)
    else:
      bot.sendPhoto(chat_id, open(outname,'rb'))
      bot.sendMessage(chat_id, userinfo[chat_id]['report'])
  #self.bot.sendMessage(chat_id, "Ok, added to the bot. Your ID is: " + str(chat_id))

bot_address = 'YOUR-ADDRESS-HERE!'
bot = telepot.Bot(bot_address)
bot.message_loop(handle)
print('I am listening ...')

while 1:
    time.sleep(1)

