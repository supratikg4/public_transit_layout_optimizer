# Public Transit Layout Optimizer

To run the code, open “transit_planner.py” in the terminal or IDE. Run the file. 

The program will prompt you to enter a numeric value that corresponds to a city map layout. The map name followed by its (width x height) dimensions are given for each input map. The input maps are shown below as follows:

1. Monocentric (10x10)

<img width="75" height="144" alt="image" src="https://github.com/user-attachments/assets/19418002-54e4-4f2c-bb5c-596f666160e6" />

2.	Twin (18x8)

<img width="132" height="115" alt="image" src="https://github.com/user-attachments/assets/f0391ed7-f728-40eb-b70e-e616e2db9326" />

3.	Dense (11x11)

<img width="87" height="160" alt="image" src="https://github.com/user-attachments/assets/19e9478e-5e45-44ec-8258-b1d85e2c1fc8" />

4.	Large (35x35)

<img width="132" height="258" alt="image" src="https://github.com/user-attachments/assets/9e750a7a-94f3-4324-b6a2-34d31e8bef20" />

5.	Manhattan (30x40)

<img width="142" height="363" alt="image" src="https://github.com/user-attachments/assets/b2da0f8b-0585-4f1a-8ab1-ffd431fd5dbb" />


The first 3 maps (1-3) run fast. Maps 4 and 5 are larger, and take significantly longer to run. After choosing the desired map, the genetic algorithm will run, displaying information about penalty violations and fitness scores as it evolves. 
After converging, it will show several plots:
1.	Convergence plot: Plot of best fitness score per generation
2.	Empty city: Commuter flow through city with no transit layout
3.	Transit city: Commuter flow through city with best layout returned from genetic algorithm, along with metrics of fitness, average commute time, and penalty violations

The other Python file, “planner_real_data.py” performs the exact same genetic algorithm, but on real street and density data provided by OpenStreetMap. To use this program, open the file in the same way as above. Run the file.
The program will prompt you to enter a “Point:”. This is the latitude, longitude point of the part of the world you want to create a transit map for. Any latitude and longitude can be used. Below are two example points that can be used. Enter this exactly as it appears in the quotations:
1.	Raleigh, NC – “35.7804, -78.6391”
2.	Manhattan Island, New York City, NY – “40.7470, -73.9887”
After inputting the coordinate point, the program will ask you to enter a size. This is the radius from the point to make a map for. Enter 2000 for a default value that does not take too long to run.
After entering the size of the zone, the program will begin to fetch data from OSMnx about road network and population density of the entered zone, which will take about 15-30 seconds, after which the genetic algorithm will run the same as the original implementation. After converging, a convergence plot will show, followed by the transit layout and commuter flow of the location entered.
