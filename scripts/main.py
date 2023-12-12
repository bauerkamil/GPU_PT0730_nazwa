import csv
import matplotlib.pyplot as plt


def read_csv(file_path):
    # Open the CSV file
    with open(file_path + '.csv', 'r') as csv_file:
        # Create a CSV reader
        csv_reader = csv.reader(csv_file, delimiter=';')

        # Read the header
        header = next(csv_reader)

        # Find the indices of 'size' and 'time' in the header
        size_index = header.index('size')
        time_index = header.index('achieved_occupancy')

        # Initialize lists to store size and time data
        data = {}

        # Read the rows and extract data
        for row in csv_reader:
            size = int(row[size_index])
            time = float(row[time_index])

            if size not in data:
                data[size] = [time]
            else:
                data[size].append(time)

        avg_data = {}
        for size, times in data.items():
            avg_data[size] = sum(times) / len(times)

    return avg_data.keys(), avg_data.values(), 'Eratosthenes' if 'outputEratosthenesNsight' in file_path else 'Sundaram'


def plot_and_save_data(pairs, plot_path):
    plt.clf()
    for sizes, times, name in pairs:
        plt.plot(sizes, times, '-', label=name)
        plt.plot(sizes, times, 'o')

    plt.xscale('log')
    plt.title('Wykres funkcji osiągniętej zajętości w zależności od rozmiaru')
    plt.xlabel('Rozmiar')
    plt.ylabel('Osiągnięta zajętość')
    plt.legend()
    plt.savefig(plot_path)


def main():
    groups = [['outputEratosthenesNsight', 'outputSundaramNsight']]

    for group in groups:
        pairs = []
        for file_name in group:
            # Read the CSV file
            pairs.append(read_csv(f'./nsight_data/{file_name}'))

        plot_and_save_data(pairs, 'achieved_occupancy.png')


if __name__ == '__main__':
    main()

