import yaml

def extract_yaml(filepath):
    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)
    
    return results

def print_lines(dict, streak_properties_list):
    str = ''
    for property in streak_properties_list:
        str += dict[property] + '   '

    return str

class PrintResults:
    def __init__(self, filepath):
        self.filepath = filepath
        self.current_dict = None
    
    def print_result_lines(self, streak_properties_list):
        print_lines = []
        results_dict = extract_yaml(self.filepath)
        self.current_dict = results_dict

        for main_file in results_dict.keys():
            string = main_file + '  '

            self.current_dict = self.current_dict[main_file]
            for sub_file in self.current_dict.keys():
                string += sub_file + '  '
                self.current_dict = self.current_dict[sub_file]
                string += print_lines(self.current_dict, streak_properties_list)

        return print_lines
            
        


            

         
