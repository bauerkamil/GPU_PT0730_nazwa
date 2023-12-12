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
        time_index = header.index('time')

        # Initialize lists to store size and time data
        data = {}

        # Read the rows and extract data
        for row in csv_reader:
            size = int(row[size_index])
            time = int(row[time_index])

            if size not in data:
                data[size] = [time]
            else:
                data[size].append(time)

        avg_data = {}
        for size, times in data.items():
            avg_data[size] = sum(times) / len(times)

    return avg_data.keys(), avg_data.values(), file_path[-3:]


def plot_and_save_data(pairs, plot_path):
    plt.clf()
    for sizes, times, name in pairs:
        plt.plot(sizes, times, '-', label=name)
        plt.plot(sizes, times, 'o')

    plt.xscale('log')
    plt.title('Wykres funkcji czasu w zależności od rozmiaru')
    plt.xlabel('Rozmiar')
    plt.ylabel('Czas')
    plt.legend()
    plt.savefig(plot_path)


def main():
    groups = [['outputEraGPU', 'outputEraCPU'], ['outputSundaGPU', 'outputSundaCPU']]

    for group in groups:
        pairs = []
        for file_name in group:
            # Read the CSV file
            pairs.append(read_csv(file_name))

        plot_and_save_data(pairs, ''.join(group)+'.png')


if __name__ == '__main__':
    main()

