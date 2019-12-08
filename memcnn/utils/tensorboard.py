import collections
import glob
import json
import os
try:
    from tensorflow.train import summary_iterator
except ImportError:
    from tensorflow.compat.v1.train import summary_iterator


def parse_logs(dir_path, output_file = None):
    # collect all log_files from dir_path
    log_files = glob.glob(os.path.join(dir_path, 'events.out.tfevents.*'))
    event_dict = {}
    # iterate all log_fiiles in order of creation
    for log_file in log_files:
        for event in summary_iterator(log_file):
            for value in event.summary.value:
                if not (value.tag in event_dict):
                    event_dict[value.tag] = collections.OrderedDict()
                if value.HasField('simple_value'):
                    event_dict[value.tag][event.step] = (event.wall_time, event.step, value.simple_value)
    # flatten ordered dicts to lists
    for key in event_dict.keys():
        event_dict[key] = list(event_dict[key].values())
    # optionally write event_dict to json
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(event_dict, f)
    return event_dict
