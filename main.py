import csv
import matplotlib.pyplot as plt

def read_csv(file_path):
    # Open the CSV file
    with open(file_path, 'r') as csv_file:
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

    return avg_data.keys(), avg_data.values()


def plot_and_save_data(sizes, times, plot_path):
    plt.clf()

    # Connect the points with a line
    plt.plot(sizes, times, '-')

    # Create a plot
    plt.plot(sizes, times, 'o')

    # Add a title
    plt.title('Wykres funkcji czasu w zależności od rozmiaru')

    # Add labels
    plt.xlabel('Rozmiar')
    plt.ylabel('Czas')

    # Save the plot
    plt.xscale('log')
    plt.savefig(plot_path)


def main():
    file_names = ['outputEraGPU', 'outputEraCPU', 'outputSundaGPU', 'outputSundaCPU']

    for file_name in file_names:
        # Read the CSV file
        sizes, times = read_csv(f'{file_name}.csv')

        # Plot and save the data
        plot_and_save_data(sizes, times, f'{file_name}.png')

if __name__ == '__main__':
    main()

