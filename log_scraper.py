


class LogScraper:
    def __init__(self):
        pass

    def read_log(self, log_path):

        with open(log_path) as f:
            lines = f.readlines()
        return lines

    def get_property(self, line, pattern, num_digits = 1 ):
        # this function receives a log line and looks for n_digits after the pattern


        # find the pattern. mind that there can be up to 3 unknown characters between the pattern and the digits
        pattern = pattern.replace(' ', r'\s*')
        pattern = pattern.replace('?', r'\w?')
        pattern = pattern.replace('(', r'\(')
        pattern = pattern.replace(')', r'\)')







