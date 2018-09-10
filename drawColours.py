# This script requires a couple of libraries to be installed and loaded
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import random
from random import randint


size = 10; # number of colours. Takes the valuew 10, 100 and 1000


permutation = np.arange(size)

# Display the colours in the order of the permutation in a pyplot window 
def plot_colours(colours, permutation):

	assert len(colours) == len(permutation)
	
	ratio = 10 # ratio of line height/width, e.g. colour lines will have height 10 and width 1

	img = np.zeros((ratio, len(colours), 3))

	for i in range(0, len(colours)):
		img[:, i, :] = colours[permutation[i]]

	fig, axes = plt.subplots(1, figsize=(8,4)) # figsize=(width,height) handles window dimensions
	axes.imshow(img, interpolation='nearest')
	axes.axis('off')
	plt.show()



#read file: Returns: n = no. colours, colours = 3 dimension array with red, green, blue coordinates
#IMPORTANT!! In order to read the file information I deleted the comments that were with "#" inside the file. The file must contain only the numbers.
def read_kfile(fname):
    with open(fname, 'rU') as kfile:
        lines = kfile.readlines()
    n = int(lines[0])   #n=1000 in this case
    colour = [map(float, line.split()) for line in lines[1:n+1]]    #list with 3 dimendion array colours.
    return n,colour
#evaluate
def evaluate(colours, sol):
    size=len(sol) -1 #size of the list with the Euclidean distances, because the maximum range is from 0 to 999 (1000 values)
    euclidean=[]
    for i in range(0,size):
        #calculates the distance between the the colours in the list
        euclidean_distance = sqrt((colours[sol[i]][0]-colours[sol[i+1]][0])**2+(colours[sol[i]][1]-colours[sol[i+1]][1])**2+(colours[sol[i]][2]-colours[sol[i+1]][2])**2)
        euclidean.append(euclidean_distance) #append all the distances in that list

    return sum(euclidean) #returns the sum of the distances between all the colours


#initalise random solution indexes of colours
def initialise(size):
    rand = random.sample(range(0, 1000), size)  #random.sample gives a list with distinct values
    #print rand
    return rand #returns a list with random distinct numbers from 0 to 1000

#calculate the mean of Euclidean distances
def mean(sol):
    num_items=len(sol)  #number of distances
    mean = sum(sol) / num_items  #the formula is the sum of the distance divided by the number of the population/sample
    return mean  #returns the mean of the distances

#calculate the standard deviation of Euclidean distances
def standard_deviation(sol, population=True):
    num_items=len(sol)
    differences = [x - mean(sol) for x in sol] #substraction of the distances and the mean
    sq_differences = [d ** 2 for d in differences]  #set the result from the substaction to the power of 2
    ssd = sum(sq_differences)   #sum of all the previous calculations
    if population is True:  #I set popolation True because if we try to find the variance of a sample the denominator of the formula is different (number of items - 1)
        variance = ssd / num_items  #the formula of variance of population
    else:
        variance = ssd / (num_items - 1) #the formula of variance of a sample
    sd = sqrt(variance)     #square route in the variance in order to find the the standard deviation
    return sd   #returns the standard deviatiopn

#random search algorithm and solutions for the given runs of the algorithm
def random_search(iterations, colours):   #gets the number of iterations that I want to commit (in that case I set 100) and the list of colours
    rand_runs=[]
    sol_runs=[]
    group_rs = []
    for i in range(iterations):
        rand1=initialise(size)  #initalise random indexes of colours
        rand_runs.append(rand1) #append every list of the indexes of colours in the list rand_runs
        sol_runs.append(evaluate(colours,rand1))    #append every sum of Euclidean distances in list sol_runs
    min_sol = sol_runs[0]   #set the first value in the list sol_runs as the best slution
    min_rand = rand_runs[0] #set the corresponding list of indexes of colours as the best solution 
    for i in range(1,iterations): #the range starts from 1 until the number of runs, because I have set the first value of the list as the best (0)
        if min_sol> sol_runs[i]:    #check if there are other better solutions
            min_sol = sol_runs[i]   #if the conitions is right change the value of mon_sol with the value that satisfies the condition
            min_rand = rand_runs[i] #change the value for the correspnding list of indexes
        group_rs.append(min_sol)    #list with the best solutions in random search in every iteration
    return sol_runs, rand_runs, min_sol, min_rand, group_rs   #returns the list of  Euclidean distance in the number of runs, the list with the indexes of the colours of every solution, the best sum of Euclidean distances of all the runs, the list of indexes of the best fitness, the list of the best solution during the iteration

# Box plot of the Euclidean distances that showd the range and the mean of the results
def plot_sol(eval_sol): #gets a list of the Euclidean distances of algorithms
#   plt.subplot(121)
   plt.boxplot(eval_sol)
   plt.title("Euclidean distances rand_search")
   plt.ylabel("Euclidean distance")
   plt.show

   

#plot_sol(random_search(20, colours)[0])

def plot_linegraph(sol_random_search, sol_hillclimber, sol_iterated_local_search, sol_genetic_algorithm):
#    plt.subplot(122)
    best_solutions = [sol_random_search, sol_hillclimber, sol_iterated_local_search, sol_genetic_algorithm]     #values in y axis
    algorithm = ["random search", "hill climb", "iterated local search", "gentic algorithm"]    #nominal values in x axis
    plt.plot(algorithm, best_solutions)
    plt.show

def line_graph(alg_sol):
    plt.title("Euclidean distances during iterations")  #the title of the line graph
    plt.xlabel('iterations')    #x axis
    plt.ylabel('Euclidean distances')   #y axis
    plt.plot(alg_sol)   #plot the list with the values that the function gets
    plt.show()

def line_graph_all(rand_search, hill_climb, it_l_s, gen_alg):   #operator that creates a line graph that we put the values from the algorithms in order to compare them
    plt.title("Euclidean distances during iterations")  #the title of the line graph
    plt.xlabel('iterations')    #x axis
    plt.ylabel('Euclidean distances')   #y axis
    plt.plot(rand_search)   #plot the list with the values that the function gets
    plt.plot(hill_climb)
    plt.plot(it_l_s)
    plt.plot(gen_alg)
    plt.legend(['Random Search', 'Hill Climb', 'Iterated Local Search', 'Genetic Algorithm'], loc='lower left')
    plt.show()

#2nd question
#swap/ Mutation operator
def swap(sol):  #get the list of indexes of the colours which comes for initialise(size) operator
    l = len(sol)    #l gets the value of the length of sol list
    pos1 = pos2 = randint(0, l-1)   #set randomly position 1 and position 2
    while (pos1 == pos2):   #in order not to have the same values for pos1 and pos2
        pos2 = randint(0, l-1)
    value= sol[pos1]    #keep track of the value in pos1
    sol[pos1] = sol[pos2]   #replace value in pos1 with pos2 value
    sol[pos2] = value   #replace value in pos2 with pos1 value
    return sol  #return the list of indexes with the swap changes

#perturbation operator
def perturbation(sol):  #get the list of indexes of the colours which comes from initialise(size) operator
    #it is the same process of swap function but here we do this twice
    l = len(sol)
    pos1 = pos2 = randint(0, l-1)
    while (pos1 == pos2):
        pos2 = randint(0, l-1)
    value= sol[pos1]
    sol[pos1] = sol[pos2]
    sol[pos2] = value
    pos3 = pos4 = randint(0, l-1)
    while (pos3 == pos4):
        pos4 = randint(0, l-1)
    value= sol[pos3]
    sol[pos3] = sol[pos4]
    sol[pos4] = value
    return sol

#inverse hill climber
def reverse(sol):   #get the list of indexes of the colours which comes for initialise(size) operator
    new_sol=[]
    l = len(sol)
    pos1 = pos2 = randint(0, l-1)   #set randomly position 1 and position 2
    while (pos1 == pos2):   #in order not to have the same values for pos1 and pos2
        pos2 = randint(0, l-1)
    #print pos1, pos2
    if pos1<pos2:   #I set this condition in order to from which position do I have o start
        value=sol[pos1] #keep track of the value in pos1
        if pos1!=0:
            for j in sol[0:pos1]:   #the values between 0 and pos1
                new_sol.append(j)   #append the values that are before pos1 in the list
            for j in sol[pos2:pos1:-1]: #inverse the values between pos1 and pos2
                new_sol.append(j)   #append the inverted values in the list
                sol[pos2]=value     #change also the pos2 value because it does not change with the above process 
        else:
            for j in sol[pos2:pos1:-1]: #if pos1 = 0  inverse all the values between pos1 and pos2. It is in the beginning of the list
                new_sol.append(j)   #append the inverted values in the list
                sol[pos2]=value     #change also the pos2 value because it does not change with the above process 
        for j in sol[pos2:l]:   #all the values after pos2
            new_sol.append(j)   # append all the values after pos2
                        
    else:   #if pos1>pos2
        #it is the same process as above but this time we have pos2 before pos1
        value=sol[pos2] 
        if pos2!=0:
            for j in sol[0:pos2]:
                new_sol.append(j)
            for j in sol[pos1:pos2:-1]:
                new_sol.append(j)
                sol[pos1]=value
        else:
            for j in sol[pos1:pos2:-1]:
                new_sol.append(j)
                sol[pos1]=value
        for j in sol[pos1:l]:
            new_sol.append(j)
    return new_sol  #returns the solution that has been inverted


# 1pt- crossover. Input: 2 lists of indexes of colours. Returns a single offspring
def xover_1pt(sol1, sol2):
    xp =randint(0,len(sol1)-1)
    return[sol1[:xp] + sol2[xp:]]    



#Hill Climber
def hillClimbingBest(sol):  #get the list of indexes of the colours which comes from initialise(size) operator

    rand1 = sol     #keep track of sol
    best = evaluate(colours,sol)    #evaluate the Euclidean distance of the list sol
    group_hill=[]
    for i in range(100):    #100 iterations
        sol=swap(sol)   #swap sol list
        solution = evaluate(colours,sol)    #evaluate the Euclidean distance of the list sol after the swap
        if solution<best:   #check if the new solution is better than the previous
            best=solution   # if the new solution is better change the best value with the value of the new solution
            rand1=sol   #change rand1 with the new sol
        group_hill.append(best) #keep track of the best solution during the iteration in grou_hill list
    return best, rand1, group_hill  #returns the best Euclidean distance of the 100 iterations, its corresponding list of indexes of colours and the list with the best solution during the iterations

#function that runs the hill climber 20 times
def hill_climb_20runs(sol): #get the list of indexes of the colours which comes from initialise(size) operator
    hc_sol=[]
    hc_rand=[]
    for j in range(20): #20 runs
         rand_hill_climb = hillClimbingBest(sol)[1] #store the list of indexes of colours of the best Euclidean distance in hill climber
         sol_hill_climb = hillClimbingBest(sol)[0]  #store the best Euclidean distance of hill climber
         hc_sol.append(sol_hill_climb)  #append the best Euclidean distance of hill climberin hc_sol list
         hc_rand.append(rand_hill_climb)    #append its corresponding lists of indexes in hc_rand list
    min_sol = hc_sol[0]     #set the first value of the hc_sol as the best
    min_rand = hc_rand[0]   #set the corresponding list of indexes of colours as the best
    for i in range(1,20):   #I start from 1 because I have set the first value as the best
        if min_sol> hc_sol[i]:  #check if any value in hc_sol is better than min_sol
            min_sol = hc_sol[i]     #if the condition is correct change min_sol to the value of the condition
            min_rand = hc_rand[i]   # change the corresponding list of indexes of colours
    return min_sol, min_rand, hc_sol, hc_rand   #returns the best Euclidean distance of the 20 runs of hill climber, its corresponding lists of indexes of colours, list with the 20 Euclidean distances of hill climber, and the 20 lists of indexes of colours
         


#iterated local search

def iterated_local_search(sol): #get the list of indexes of the colours which comes for initialise(size) operator

        iter_100 = []
        iter_100_rand = []
        best = hillClimbingBest(sol)[1] #store the list of indexes of colours of the best Euclidean distance in hill climber
        best_sol = hillClimbingBest(sol)[0] #store the best Euclidean distance of hill climber
        for j in range(100):    #100 iterations
            sol = perturbation(sol) #perturbate the list of indexes of colours that the function gets
            sol = hillClimbingBest(sol)[1]  #implement hill climbing to the perturbated list
            sol1 = evaluate(colours,sol)    #evaluate the Euclidean distance of the list of colours after perturbation and hill climbing
            if sol1<best_sol:   #check if sol1 is better than best_sol
                best= sol   #if the condition is correct change best with sol
                best_sol = sol1     #if the condition is correct change best_sol with sol1
            iter_100.append(best_sol)   #append the 100 best_sol Euclidean distances that we get
            iter_100_rand.append(best)  #append their corresponding list of indexes of colours
        best2 = iter_100[0]     #set the first value of the iter_100 as the best
        best2_rand = iter_100_rand[0]   #set the corresponding list of indexes of colours as the best
        for i in range(1,100):  #I start from 1 because I have set the first value as the best
            if best2>iter_100[i]:    #check if any value in iter_100 is better than best2
                best2 = iter_100[i]     #if the condition is correct change best2 to the value of that satisfies the condition
                best2_rand = iter_100_rand[i]   # change the corresponding list of indexes of colours

        return best2, best2_rand, iter_100, iter_100_rand   #returns the best Euclidean distance of the 100 iterations of iterated local search, its corresponding list of indexes of colours, list with the 100 Euclidean distances of iterated local search, and the 100 lists of indexes of colours


#Genetic Algorithm
def genetic_alg(population, size1): #gets the size of population and the number of colours
    
        group = []
        group_rand=[]
        group_100 = []
        group_100_rand = []
        for i in range(population): #loop in order to create a population
            rand = initialise(size1)    #initialise a list of random index of colours
            solution = evaluate(colours,rand)   #evaluate the euclidean distance of the solution from initialse(size)
            group.append(solution)  #append the euclidean distances of that are found
            group_rand.append(rand) #append the list of random index of colours
            #the group and group_rand constitutes the population
        #Tournament Selection
        for j in range(100):    #100 iterations
            ind1 = random.randint(0,population-1) #select_random_individual(P)
            ind2 = random.randint(0,population-1) #select_random_individual(P)
            
            if group[ind1]<group[ind2]: #I choose the best Euclidean distance in order to create a mom
                #mom = group[ind1]
                mom_rand = group_rand[ind1] #the list of random index of colours of the best mom
            else:
                # mom = group[ind2]
                mom_rand = group_rand[ind2] #the list of random index of colours of the best mom
            ind3 = random.randint(0,population-1)   #select_random_individual(P)
            ind4 = random.randint(0,population-1)   #select_random_individual(P)
            if group[ind3]<group[ind4]:     #I choose the best Euclidean distance in order to create a dad
                #dad = group[ind3]
                dad_rand = group_rand[ind3] #the list of random index of colours of the best dad
            else:
                # dad = group[ind4]
                dad_rand = group_rand[ind4] #the list of random index of colours of the best dad
                #print mom_rand, dad_rand, 
                
    
            #create the two children
            child1 = xover_1pt(mom_rand, dad_rand)
            child2 = xover_1pt(mom_rand, dad_rand)
            
            #I mutate the childen with the swap operator
            mutation_rate = random.randint(0,100)   #mutation_rate is a random integer, which expresses percentage, and is the probability rate that get in order to mutate the children
            if mutation_rate <= 10: #set the probability rate = 10%
                mut_child1= swap(child1[0]) #I put [0] to child because it is an embedded list
                mut_child2= swap(child2[0])
            else:   #else I do not mutate the children
                mut_child1 = child1[0]
                mut_child2 = child2[0]
           
            #Now I will find the two worse Euclidean distances in the population

            index1=0    #I keep track of the index of the worse value. I set as max value (of Euclidean distances) the first value in the population
            index2=1    #I keep track of the index of the 2nd worse value. I set as 2nd max value (of Euclidean distances) the second value in the population
            for i in range(2,population):   #I start checking from 2 because I have set the two first values as the worse
                if group[i]>group[index1] : #if any value in the population is worse
                    index1=i #change the index to that
                elif group[i]>group[index2]:    #if any other value is worse than the 2nd worse value
                    index2=i #change that value index

            group[index1]=evaluate(colours, mut_child1) #replace the 1st worse value with the Euclidean distance of the first children
            group_rand[index1]=mut_child1   #replce the list of indexes of colours with child1

            group[index2]=evaluate(colours, mut_child2) #replace the 2nd worse value with the Euclidean distance of the 2nd children
            group_rand[index2]=mut_child2   #replce the list of indexes of colours with child2
    
            #print "The first value that is going to be replaced is: ",index1, " and the 2nd: ", index2
            #print max_value1, max_rand1
            #print max_value2, max_rand2
    
            #I will find the best of all the values
            best=group[0] #set the first value of the list as the best
            best_rand=group_rand[0] #the corresponding list of indexes of colours
            
            for i in range(1,population):   #I start from 1 because I have set the first value as the best
                if best>group[i]:   #check if any value in group is less than the best
                    best=group[i]   #if the condition is correct change the best value to the value of the condition
                    best_rand=group_rand[i] # change the corresponding list of indexes of colours
            group_100.append(best)  #append all the 100 best values of genetic algorithm in group_100 list 
            group_100_rand.append(best_rand)    #append all the best list of indexes of colours of genetic algorithm, in group_100_rand
        best2 = group_100[0]    #set the first value of the group_100 as the best
        best2_rand = group_100_rand[0]  #set the corresponding list of indexes of colours as the best
        for i in range(1,100):  #I start from 1 because I have set the first value as the best
            if best2>group_100[i]:  #check if any value in group_100 is better than the best
                best2 = group_100[i] #if the condition is correct change the best value to the value of the condition
                best2_rand = group_100_rand[i]  # change the corresponding list of indexes of colours
       
        return best2, best2_rand, group_100, group_100_rand     #returns the best solutions, the corresponding list of indexes of colours, a list of the best solutions in every iteration and the coresponding lists of indexes of colours

###################################################################################################################
n, colours = read_kfile('colours.txt')  #read n=1000 and the colours from the file
rand=initialise(size)   #initialise randomly a list with colour indexes

##################################################################################################################
#20 runs of Random Search Algorithm

rs_20 =[]
rs_20_rand = []
for i in range(20): #run for 20 times
    rs = random_search(100, colours) #store to variable rs the result of random_search function, that does 100 iterations
    rs_20.append(rs[2])   #append the best Euclidean distance from genetic algorithm
    rs_20_rand.append(rs[3])  #append the corresponding list of indexes of colours
best_rs = rs_20[0]    #set the first value of the gen_20 as the best
best_rs_rand = rs_20_rand[0]  #set the corresponding list of indexes of colours as the best
for i in range(1,20):   #I start from 1 because I have set the first value as the best
    if best_rs > rs_20[i]:    #check if any value in gen_20 is better than the best_it
        best_rs = rs_20[i]    #if the condition is correct change the best value to the value that satisfies the condition
        best_rs_rand = rs_20_rand[i]  #change the corresponding list of indexes of colours
print "The 20 Euclidean distances in the random search algorithm are: ", rs_20
print "The mean of 20 Euclidean distances in the random search algorithm is: ", mean(rs_20)
print "The standard deviation of the 20 Euclidean distances in the random search algorithm is: ", standard_deviation(rs_20, population=True)
print "The best Euclidean distance is: ", best_rs
plot_sol(rs_20)   #Box plot of the 20 Euclidean distances of random search results

colours1 = []
for i in range(0, size):
    colours1.append(colours[best_rs_rand[i]])  #rs[3] is the list with the indexes of the colours of best solution
plot_colours(colours1, permutation)

#line_graph(random_search(100, colours)[4])     #line graph of the euclidean distances during the iterations in random search

##############################################################################################################
#20 runs of Hill Climb Algorithm
hc = hill_climb_20runs(rand)    #store to variable hc the result of hill_climb_20runs function
print "The 20 Euclidean distances in the hill climb algorithm are: ", hc[2]
print "The mean of 20 Euclidean distances in the hill climb algorithm is: ", mean(hc[2])
print "The standard deviation of the 20 Euclidean distances in the hill climb algorithm is: ", standard_deviation(hc[2], population=True)
print "The best Euclidean distance is: ", hc[0]
plot_sol(hc[2])   #Box plot of the 20 Euclidean distances of hill climber results

colours2 = []
for i in range(0, size):
    colours2.append(colours[hc[1][i]])  #hc[1] is the list with the indexes of the colours of best solution
plot_colours(colours2, permutation)

#line_graph(hillClimbingBest(rand)[2])  #line graph of the euclidean distances during the iterations of hill climber
###############################################################################################
#20 runs of Iterated Local Search Algorithm
iter_20 = []
iter_20_rand =[]
for i in range(20): #run for 20 times
    it = iterated_local_search(rand)    #store to variable it the result of iterated_local_search function
    iter_20.append(it[0])   #append the best Euclidean distance from iterated local search
    iter_20_rand.append(it[1])  #append the corresponding list of indexes of colours
best_it = iter_20[0]    #set the first value of the iter_20 as the best
best_it_rand = iter_20_rand[0]  #set the corresponding list of indexes of colours as the best
for i in range(1,20):   #I start from 1 because I have set the first value as the best
    if best_it > iter_20[i]:    #check if any value in iter_20 is better than the best_it
        best_it = iter_20[i]    #if the condition is correct change the best value to the value that satisfies the condition
        best_it_rand = iter_20_rand[i]  #change the corresponding list of indexes of colours
print "The 20 Euclidean distances in the iterated local search algorithm are: ", iter_20
print "The mean of 20 Euclidean distances in the iterated local search algorithm is: ", mean(iter_20)
print "The standard deviation of the 20 Euclidean distances in the iterated local search algorithm is: ", standard_deviation(iter_20, population=True)
print "The best Euclidean distance is: ", best_it
plot_sol(iter_20)   #Box plot of the 20 Euclidean distances of iterated local search results

colours3 = []
for i in range(0, size):
    colours3.append(colours[best_it_rand[i]])   #best_it_rand is the list with the indexes of the colours of best solution
plot_colours(colours3, permutation)

#line_graph(it[2])    #line graph of Euclidean distance in  iterated local search during the iterations

###########################################################################################
#20 runs of Genetic algirithm
gen_20 =[]
gen_20_rand = []
for i in range(20): #run for 20 times
    gen = genetic_alg(50,size) #store to variable gen the result of genetic_alg with population 50 and size of colours equal with the size that we set in the beginning
    gen_20.append(gen[0])   #append the best Euclidean distance from genetic algorithm
    gen_20_rand.append(gen[1])  #append the corresponding list of indexes of colours
best_gen = gen_20[0]    #set the first value of the gen_20 as the best
best_gen_rand = gen_20_rand[0]  #set the corresponding list of indexes of colours as the best
for i in range(1,20):   #I start from 1 because I have set the first value as the best
    if best_gen > gen_20[i]:    #check if any value in gen_20 is better than the best_it
        best_gen = gen_20[i]    #if the condition is correct change the best value to the value that satisfies the condition
        best_gen_rand = gen_20_rand[i]  #change the corresponding list of indexes of colours
print "The 20 Euclidean distances in the genetic algorithm are: ", gen_20
print "The mean of 20 Euclidean distances in the genetic algorithm is: ", mean(gen_20)
print "The standard deviation of the 20 Euclidean distances in the genetic algorithm is: ", standard_deviation(gen_20, population=True)
print "The best Euclidean distance is: ", best_gen
plot_sol(gen_20)   #Box plot of the 20 Euclidean distances of genetic algorithm results

#plot the colours of the best result
colours4 = []
for i in range(0, size):
    colours4.append(colours[best_gen_rand[i]])  #best_gen_rand is the list with the indexes of the colours of best solution
plot_colours(colours4, permutation)

#line_graph(gen[2])     #line graph of Euclidean distance in a genetic algorithm during the iterations

#####################################################################################################################

line_graph_all(random_search(100, colours)[4],hillClimbingBest(rand)[2],it[2],gen[2])      #line graph that compares the algorithms