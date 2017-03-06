import sys
import time
import json
import subprocess
import os
import platform
import psutil
import telepot
import signal
import re

signal.siginterrupt(signal.SIGHUP, False)

class Chat(object):

    def __init__(self, msg):
        self.id = msg.get('id', None)
        self.username = msg.get('username', None)
        self.first_name = msg.get('first_name', None)
        self.type = msg.get('type', None)


class From(object):

    def __init__(self, msg):
        self.id = msg.get('id', None)
        self.username = msg.get('username', None)
        self.first_name = msg.get('first_name', None)


class TMsg(object):

    def __init__(self, msg):
        self.text = msg.get('text', None)
        self.date = msg.get('date', None)
        self.message_id = msg.get('message_id', None)
        self.chat = Chat(msg['chat'])
        self.from_ = From(msg['from'])


class TFBot(object):

    def __init__(self):
        with open("config.json", "r") as config_file:
            self.config = json.load(config_file)

        self.bot = telepot.Bot(self.config['TOKEN'])
        self.bot.message_loop({
            'chat': self.on_chat_message
        })
        self.__bash_timeout = 60

        self.__tfenv = """export WORKON_HOME=$HOME/.virtualenvs &&
export PROJECT_HOME=$HOME/Devel &&
source /usr/local/bin/virtualenvwrapper.sh && 
workon TensorFlow && """

    def __bash_call(self, msg, bash_cmd, ok_msg):
        proc = subprocess.Popen(
            bash_cmd,
            cwd=self.config['project_dir'],
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable='bash'
        )
        outs = None
        errs = None
        try:
            outs, errs = proc.communicate(timeout=self.__bash_timeout)
        except subprocess.TimeoutExpired:
            self.bot.sendMessage(
                msg.chat.id,
                "Warning, timeout occurred, process will be killed..."
            )
            proc.kill()
            outs, errs = proc.communicate()
        else:
            if proc.returncode == 0:
                self.bot.sendMessage(
                    msg.chat.id,
                    ok_msg
                )
                return True, outs
            else:
                self.bot.sendMessage(
                    msg.chat.id,
                    "Some error occurred during execution of '{}'".format(
                        bash_cmd)
                )
                return False, errs

    @staticmethod
    def __ls_filter(res):
        files = res.decode("utf-8").split("\n")[1:]
        files = ["- " + file_.split(" ")[-1]
                 for file_ in reversed(files) if file_ != '']
        return "\n".join(files) if len(files) > 0 else "Folder empty..."

    def parse_message(self, msg):
        if msg.text == "help":
            self.bot.sendMessage(
                msg.chat.id,
                "List of command supported:\n"
                "- status (Show CPU and RAM status)\n"
                "- timeout n (set timeout for single operation)\n"
                "- make clean (clean compiled lib)\n"
                "- make (make libs)\n"
                "- results (show results folder)\n"
                "- configs (show config files)\n"
                "- logs (show log files)\n"
                "- dataests (show datasets)\n"
                "- zip name (create a zip of results and send it)\n"
                "- run name (run a configured DENN, only file name)\n"
                "- cat log_name (cat on a log file)\n"
                "- tail log_name (tail on a log file)\n"
                "- update (update all git repo and reset it)\n"
                "- killall (kill all python processes)\n"
            )
        elif msg.text == "status":
            cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
            v_mem = psutil.virtual_memory()
            s_mem = psutil.swap_memory()
            self.bot.sendMessage(
                msg.chat.id,
                "+ CPU usage: {}".format(str(cpu_usage)) +
                "\n+ Ram: {}".format(v_mem.percent) +
                "\n+ Swap: {}".format(s_mem.percent)
            )
        elif msg.text == "killall":
            for proc in psutil.process_iter():
                try:
                    pinfo = proc.as_dict(attrs=['pid', 'name', 'cmdline'])
                except psutil.NoSuchProcess:
                    pass
                else:
                    if pinfo['name'].lower() == "python" and\
                            "tf_bot.py" not in pinfo['cmdline']:
                        self.__bash_call(
                            msg,
                            "kill -9 {}".format(pinfo['pid']),
                            "+ killed python processes with pid {}!".format(pinfo[
                                                                            'pid'])
                        )
        elif msg.text.find("timeout") != -1:
            try:
                _, timeout = msg.text.split(" ")
                self.__bash_timeout = int(timeout.strip())
                self.bot.sendMessage(
                    msg.chat.id,
                    "Set timout {}".format(self.__bash_timeout)
                )
            except:
                self.bot.sendMessage(
                    msg.chat.id,
                    "Can't set timeout, wrong command!"
                )
        elif msg.text == "update":
            self.__bash_call(
                msg,
                "git reset --hard HEAD",
                "+ git reset HEAD"
            )
            self.__bash_call(
                msg,
                "git pull",
                "+ git pull"
            )
            self.__bash_call(
                msg,
                "cd datasets && git pull origin master",
                "+ git pull datasets"
            )
        elif msg.text == "make clean":
            self.__bash_call(
                msg,
                "make clean",
                "Clean done!"
            )
        elif msg.text == "make":
            self.__bash_call(
                msg,
                self.__tfenv + "make",
                "Made all libs!"
            )
        elif msg.text == "results" or\
                msg.text == "configs" or\
                msg.text == "datasets" or\
                msg.text == "logs":
            ls_folder = None
            if msg.text == "results":
                ls_folder = "ls -ltr ./scripts/benchmark_results"
            elif msg.text == "configs":
                ls_folder = "ls -ltr ./scripts/config/*.json"
            elif msg.text == "logs":
                ls_folder = "ls -ltr ./scripts/logs/*.out"
            elif msg.text == "datasets":
                ls_folder = "ls -ltr ./datasets/*.gz"
            op_ret, res = self.__bash_call(
                msg,
                ls_folder,
                "Folder[{}]".format(msg.text)
            )
            self.bot.sendMessage(
                msg.chat.id,
                self.__ls_filter(res)
            )
        elif msg.text.find("cat") != -1 or\
                msg.text.find("tail") != -1:
            try:
                command, filename = msg.text.split(" ")
                is_tail = command.find('tail') != -1
                command =  command + ' -n 2' if is_tail else command
                op_ret, res = self.__bash_call(
                    msg,
                    "cd ./scripts/logs && {} {}".format(
                        command,
                        filename
                    ),
                    "+ FILE {} with {}".format(filename, command)
                )
                if op_ret:
                    text = (res+b'\0').decode('utf-8',errors='ignore')
                    text = re.sub(r"([^\r]*)\r","",text) if is_tail else text
                    #print(text)
                    self.bot.sendMessage(
                        msg.chat.id,
                        text.encode('utf-8') if len(res) != 0 else "Empty file..."
                    )
                    #self.bot.sendDocument(msg.chat.id, document=text.encode('utf-8'))
            except:
                self.bot.sendMessage(
                    msg.chat.id,
                    "Can't use cat or tail!"
                )
                raise
        elif msg.text.find("zip") != -1:
            try:
                _, filename = msg.text.split(" ")
                try:
                    os.makedirs("./scripts/benchmark_results/bot_zip")
                except OSError as err:
                    if err.errno == 17:
                        pass
                    else:
                        raise
                op_ret, res = self.__bash_call(
                    msg,
                    "cd ./scripts/benchmark_results && rm -f -R bot_zip/{0}.zip && zip -r bot_zip/{0}.zip {0}".format(
                        filename
                    ),
                    "Result zipped!"
                )
                if op_ret:
                    with open("../benchmark_results/bot_zip/{0}.zip".format(filename), "rb") as zip_file:
                        self.bot.sendDocument(
                            msg.chat.id,
                            zip_file
                        )
            except:
                self.bot.sendMessage(
                    msg.chat.id,
                    "Can't zip results!"
                )
        elif msg.text.find("run") != -1:
            try:
                _, filename = msg.text.split(" ")
                op_ret, res = self.__bash_call(
                    msg,
                    self.__tfenv + "cd ./scripts && ./nohup_launch.sh benchmark.py config/{}".format(
                        filename
                    ),
                    "+ Task launched!"
                )
            except:
                self.bot.sendMessage(
                    msg.chat.id,
                    "Can't run job!"
                )
        else:
            self.bot.sendMessage(
                msg.chat.id,
                "Wrong command, say 'help' if you have doubts..."
            )

    def on_chat_message(self, msg):
        cur_msg = TMsg(msg)
        # print(msg)

        if cur_msg.from_.id not in self.config['valid_users'] or\
                cur_msg.chat.type != 'private':
            self.bot.leaveChat(cur_msg.chat.id)
        else:
            self.parse_message(cur_msg)


def main():
    bot = TFBot()
    print('Listening ...')
    # Keep the program running.
    while 1:
        time.sleep(1000)

if __name__ == '__main__':
    main()
