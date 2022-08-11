import yaml
import csv
import pandas as pd


def extract_yaml(filepath):
    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)
    
    return results

def print_lines(properties_dict, streak_properties_list):
    properties_list = []
    for property in streak_properties_list:
        properties_list.append(str(round(properties_dict[property],2)))

    return properties_list

def print_table(results_dict, streak_properties_list):
    print_list = []

    for main_file in results_dict.keys():
        main_file_list = []
        #string = main_file + '  '
        main_file_list.append(main_file)

        for sub_file in results_dict[main_file].keys():
            #string += sub_file + '  '
            main_file_list.append(sub_file)

            for streak_id in results_dict[main_file][sub_file].keys():
                #string += str(streak_id) + '    '
                main_file_list.append(str(streak_id))
                properties_list = print_lines(results_dict[main_file][sub_file][streak_id], streak_properties_list)

                for property in properties_list:
                    main_file_list.append(property)
        
        print_list.append(main_file_list)
    
    return print_list

def write_outfile(outfile_path, results_table, header):
    with open (outfile_path, 'w') as f:
        writer = csv.writer(f)

        writer.writerow(header)
        for result_line in results_table:
            writer.writerow(result_line)

def convert_to_csv(text_file, csv_outpath):
    df = pd.read_csv(text_file,delimiter=',')
    df.to_csv(csv_outpath)


if __name__ == '__main__':
    filepath = 'results.yaml'
    streak_properties_list = ['amplitude','mean_brightness','sigma','fwhm']

    results_dict = extract_yaml(filepath)
    header = ['main_file', 'sub_file', 'streak_id']
    for property in streak_properties_list:
        header.append(property)

    '''
    header_string = ''
    for h in header:
        header_string += h + ','
    '''

    final_results_list = print_table(results_dict, streak_properties_list)
    outfile = 'results_table.csv'

    write_outfile(outfile, final_results_list, header)
    convert_to_csv(outfile, 'results_formatted.csv')

    




        


            

         
