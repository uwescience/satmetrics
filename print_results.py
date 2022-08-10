import yaml

def extract_yaml(filepath):
    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)
    
    return results

def print_lines(properties_dict, streak_properties_list):
    properties_string = ''
    for property in streak_properties_list:
        properties_string += str(round(properties_dict[property],2)) + '   '

    return properties_string

def print_table(results_dict, streak_properties_list):
    print_list = []

    for main_file in results_dict.keys():
        string = main_file + '  '

        for sub_file in results_dict[main_file].keys():
            string += sub_file + '  '

            for streak_id in results_dict[main_file][sub_file].keys():
                string += str(streak_id) + '    '
                string += print_lines(results_dict[main_file][sub_file][streak_id], streak_properties_list)
        
        print_list.append(string)
    
    return print_list

def write_outfile(outfile_path, results_table, header):
    with open (outfile_path, 'w') as f:
        f.write("%s\n" % header)
        for result_line in results_table:
            f.write("%s\n" % result_line)
    

if __name__ == '__main__':
    filepath = 'test_results.yaml'
    streak_properties_list = ['amplitude','mean_brightness','sigma','fwhm']

    results_dict = extract_yaml(filepath)
    header = ['main_file', 'sub_file', 'streak_id']
    for property in streak_properties_list:
        header.append(property)

    header_string = ''
    for h in header:
        header_string += h + '      '
    
    final_results_list = print_table(results_dict, streak_properties_list)
    outfile = 'results_table.txt'

    write_outfile(outfile, final_results_list, header_string)


    




        


            

         
