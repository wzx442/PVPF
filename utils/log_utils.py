import sys
import os
import numpy as np
from datetime import datetime
from contextlib import contextmanager

class Logger:
    def __init__(self, log_file_path: str):
        """Initialize Logger, set output to file and terminal.
        
        Args:
            log_file_path (str): log file path
        """
        self.terminal = sys.stdout
        self.log_dir = os.path.dirname(log_file_path)
        
        # if directory does not exist, create it
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log = open(log_file_path, "a", encoding='utf-8')
        
        # save original NumPy print options
        self.original_np_options = {
            'threshold': np.get_printoptions()['threshold'],
            'linewidth': np.get_printoptions()['linewidth'],
            'precision': np.get_printoptions()['precision']
        }
        
        # write start time
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_initial(f"\n{'='*50}\nLog started at: {start_time}\n{'='*50}\n")

    def _write_initial(self, message):
        """Simple write method used during initialization."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def write(self, message):
        """Write message to both terminal and file."""
        # terminal uses default print options
        self.terminal.write(message)
        
        # check if message needs special handling
        if not isinstance(message, str):
            self.log.write(message)
            self.log.flush()
            return
            
        # file uses full print options
        with self.full_numpy_precision():
            # check if message contains NumPy array output
            if ('array(' in message and '...' in message) or ('ndarray' in message and '...' in message):
                try:
                    # check if message is dictionary format
                    if '{' in message and '}' in message and ':' in message:
                        # extract dictionary part
                        prefix = message[:message.find('{')]
                        dict_str = message[message.find('{'):message.rfind('}')+1]
                        suffix = message[message.rfind('}')+1:]
                        
                        # record original message as backup
                        self.log.write(f"# original output: {message}\n")
                        
                        # try to parse but do not execute
                        self.log.write(f"{prefix}dictionary content too long, please refer to the complete content part of the specific object{suffix}\n")
                    else:
                        # if not dictionary format, try to extract array part
                        if '[' in message and ']' in message and '...' in message:
                            # record original message as backup
                            self.log.write(f"# original output: {message}\n")
                            self.log.write("# array content too long, please refer to the complete content part of the specific object\n")
                        else:
                            self.log.write(message)
                except:
                    # use original message when parsing fails
                    self.log.write(message)
            else:
                # normal message without ellipsis
                self.log.write(message)
        
        self.log.flush()

    def flush(self):
        """Flush output."""
        self.terminal.flush()
        self.log.flush()

    @contextmanager
    def full_numpy_precision(self):
        """Temporarily set NumPy print options to full display."""
        original_options = np.get_printoptions()
        try:
            np.set_printoptions(
                threshold=np.inf,       # display all elements
                linewidth=np.inf,       # do not wrap
                precision=8,            # keep 8 decimal places
                suppress=True           # suppress scientific notation
            )
            yield
        finally:
            # restore
            np.set_printoptions(**original_options)

def setup_logger(log_file_name: str = "output.txt"):
    """Set up logger.
    
    Args:
        log_file_name (str): log file name, default is output.txt
        
    Returns:
        original stdout, used to restore output
    """
    # create logs directory (if not exist)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # use timestamp to create log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{timestamp}_{log_file_name}")
    
    # save original stdout
    original_stdout = sys.stdout
    
    # set new Logger
    sys.stdout = Logger(log_file_path)
    
    return original_stdout

def restore_stdout(original_stdout):
    """Restore original stdout output.
    
    Args:
        original_stdout: original stdout object
    """
    if hasattr(sys.stdout, "log"):
        sys.stdout.log.close()
    sys.stdout = original_stdout 