
import pandas as pd
import numpy as np
import random
import math

#funcao cabecalho para ordenamento da matriz geral
def delivery_points():
  return ["DP1", "DP2", "DP3", "DP4", "DP5", "DP6", "DP7", "DP8", "DP9", 
          "DP10", "DP11", "DP12", "DP13", "DP14", "DP15", "DP16"]

#funcao que integra cabecalho + matriz de valores do enunciado do problema
def distance_matrix():
  return pd.DataFrame([
    [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
    [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
    [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
    [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
    [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
    [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
    [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
    [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
    [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
    [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
    [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
    [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
    [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
    [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
    [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
    [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
    [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
  ], 
  columns=["DP0"] + delivery_points(),
  index=["DP0"] + delivery_points()
)

# hiperparâmetros
tamanho_populacao = 1000
tx_mutacao = 0.0001 #10 ** -4
tx_crossover = 0.001 #10 ** -3
tx_tragedia = 0.05
geracoes_max = 100_000
geracoes_tragedia = 1000

#rrtorna matriz para main()  
matrix = distance_matrix()
#print(matrix)

#matrix sem cabecalho para facilitar no cauculo de menor distancia
matrix_dp0 = [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662]
matrix_dp1 = [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210]
matrix_dp2 = [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754]
matrix_dp3 = [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358]
matrix_dp4 = [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244]
matrix_dp5 = [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708]
matrix_dp6 = [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480]
matrix_dp7 = [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856]
matrix_dp8 = [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514]
matrix_dp9 = [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468]
matrix_dp10 = [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354]
matrix_dp11 = [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844]
matrix_dp12 = [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730]
matrix_dp13 = [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536]
matrix_dp14 = [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194]
matrix_dp15 = [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798]
matrix_dp16 = [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0]

#inicia-se variaveis de backup com valor 1000 para uso de cauculo de menor distancia
backupitem = 1000
backupindex = 1000

#guarda-se numa lista as matrizes sem cabecalho para uso na funcao de menor distancia
lista = [matrix_dp0, matrix_dp1, matrix_dp2, matrix_dp3, matrix_dp4, matrix_dp5, matrix_dp6, matrix_dp7, matrix_dp8, 
        matrix_dp9, matrix_dp10, matrix_dp11, matrix_dp12, matrix_dp13, matrix_dp14, matrix_dp15, matrix_dp16]

#variaveis a serem utilizadas
dist_van1 = [0]
van1 = []
dist_van2 = [0]
van2 = []
dist_van3 = [0]
van3 = []
dist_van4 = [0]
van4 = []
endereco = 0
contador = 0

#VAN 1

#funcao que acha menor valor para van
while contador < 4:

  for index, item in enumerate(lista[endereco]):
    if item == 0 or index == 0 or item in van1 : item = 1000
    if item < backupitem or index < backupindex:
      menor = item
      endereco = index
    else:
      menor = backupitem
      endereco = backupindex

    backupitem = menor
    backupindex = endereco

  dist_van1.append(endereco)
  van1.append(menor)
  contador = contador + 1

dist_van1.append(0)
print("\n")
print("VAN 1  => ", van1)
print("CAMINHO VAN 1 =>" ,dist_van1)

endereco = 0
contador = 0

#VAN 2

#funcao que acha menor valor para van
while contador < 4:

  for index, item in enumerate(lista[endereco]):
    if item == 0 or index == 0 or index in dist_van1 or item in van1 or item in van2 : item = 1000
    if item < backupitem or index < backupindex:
      menor = item
      endereco = index
    else:
      menor = backupitem
      endereco = backupindex

    backupitem = menor
    backupindex = endereco

  dist_van2.append(endereco)
  van2.append(menor)
  contador = contador + 1

dist_van2.append(0)
print("\n")
print("VAN 2  => ", van2)
print("CAMINHO VAN 2 =>" ,dist_van2)

endereco = 0
contador = 0

#VAN 3

#funcao que acha menor valor para van
while contador < 4:

  for index, item in enumerate(lista[endereco]):
    if item == 0 or index == 0 or index in dist_van1 or index in dist_van2 or item in van1 or item in van2 or item in van3 : item = 1000
    if item < backupitem or index < backupindex:
      menor = item
      endereco = index
    else:
      menor = backupitem
      endereco = backupindex

    backupitem = menor
    backupindex = endereco

  dist_van3.append(endereco)
  van3.append(menor)
  contador = contador + 1

dist_van3.append(0)
print("\n")
print("VAN 3  => ", van3)
print("CAMINHO VAN 3 =>" ,dist_van3)

endereco = 0
contador = 0

#VAN 4

#funcao que acha menor valor para van
while contador < 4:

  for index, item in enumerate(lista[endereco]):
    if item == 0 or index == 0 or index in dist_van1 or index in dist_van2 or index in dist_van3 or item in van1 or item in van2 or item in van3  or item in van4 : item = 1358
    if item < backupitem or index < backupindex:
      menor = item
      endereco = index
    else:
      menor = backupitem
      endereco = backupindex

    backupitem = menor
    backupindex = endereco

  dist_van4.append(endereco)
  van4.append(menor)
  contador = contador + 1

dist_van4.append(4)
print("\n")
print("VAN 4  => ", van4)
print("CAMINHO VAN 4 =>" ,dist_van4)


#FUNCOES VAN1

def gerar_individuo():
  resultado = []
  for i in range(len(van1)):
    k = random.randint(0,16)
    j = random.randint(0,16)
    str1 = str("DP" + str(k))
    str2 = str("DP" + str(j))
    randomica = matrix[str1][str2]
    resultado.append(randomica)
    #print(resultado)
  return resultado

def fitness(senha):
  score = 0
  for i in range(len(van1)):
    if van1[i] == senha[i]: score += 0.25
  return score

# retorna populuacao mutada com uma taxa
def mutacao(populacao):
  populacao_escolhida = random.choices(populacao, k=math.ceil(tx_mutacao*len(populacao)))
  #print(populacao_escolhida)
  return [mutacao_flip(individuo) for individuo in populacao_escolhida]

# flip de valor de gene de um gene aleatório
def mutacao_flip(individuo):
  novo_individuo = list(individuo)
  index = random.randint(0, len(individuo) - 1)
  k = random.randint(0,16)
  j = random.randint(0,16)
  str1 = str("DP" + str(k))
  str2 = str("DP" + str(j))
  nova_matrix = matrix[str1][str2]
  novo_individuo[index] = nova_matrix # mutando gene
  #print(novo_individuo)
  return novo_individuo

def crossover(populacao, geracao):
  funcao_decaimento_crossover = 1 #math.exp(-geracao/200)
  qtd = funcao_decaimento_crossover*tx_crossover*len(populacao)
  populacao_crossover = []
  populacao_escolhida = random.choices(populacao, k=math.ceil(qtd))
  [1, 2, 3, 4]
  for i in range(len(populacao_escolhida) - 1):
    for j in range(i+1, len(populacao_escolhida)):
      ind1 = populacao_escolhida[i]
      ind2 = populacao_escolhida[j]

      index = random.randint(0, len(van1) - 1)
      populacao_crossover.append(ind1[0:index] + ind2[index:])
      populacao_crossover.append(ind2[0:index] + ind1[index:])

  return populacao_crossover

def selecao(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness, reverse=True)
  return nova_populacao[0:tamanho_populacao]

# escolhe os indivíduos mais aptos
def selecao_com_tragedia(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness, reverse=True)
  if (geracao % geracoes_tragedia == 0):
    tamanho_tragedia = math.ceil(tamanho_populacao*tx_tragedia)
    novos_individuos = [gerar_individuo() for _ in range(0, tamanho_populacao - tamanho_tragedia)]
    return nova_populacao[0:tamanho_tragedia] + novos_individuos
  else:
    return nova_populacao[0:tamanho_populacao]


#FUNCOES VAN2

def gerar_individuo2():
  resultado = []
  for i in range(len(van2)):
    k = random.randint(0,16)
    j = random.randint(0,16)
    str1 = str("DP" + str(k))
    str2 = str("DP" + str(j))
    randomica = matrix[str1][str2]
    resultado.append(randomica)
    #print(resultado)
  return resultado

def fitness2(senha):
  score = 0
  for i in range(len(van2)):
    if van2[i] == senha[i]: score += 0.25
  return score

# retorna populuacao mutada com uma taxa
def mutacao2(populacao):
  populacao_escolhida = random.choices(populacao, k=math.ceil(tx_mutacao*len(populacao)))
  #print(populacao_escolhida)
  return [mutacao_flip2(individuo) for individuo in populacao_escolhida]

# flip de valor de gene de um gene aleatório
def mutacao_flip2(individuo):
  novo_individuo = list(individuo)
  index = random.randint(0, len(individuo) - 1)
  k = random.randint(0,16)
  j = random.randint(0,16)
  str1 = str("DP" + str(k))
  str2 = str("DP" + str(j))
  nova_matrix = matrix[str1][str2]
  novo_individuo[index] = nova_matrix # mutando gene
  #print(novo_individuo)
  return novo_individuo

def crossover2(populacao, geracao):
  funcao_decaimento_crossover = 1 #math.exp(-geracao/200)
  qtd = funcao_decaimento_crossover*tx_crossover*len(populacao)
  populacao_crossover = []
  populacao_escolhida = random.choices(populacao, k=math.ceil(qtd))
  [1, 2, 3, 4]
  for i in range(len(populacao_escolhida) - 1):
    for j in range(i+1, len(populacao_escolhida)):
      ind1 = populacao_escolhida[i]
      ind2 = populacao_escolhida[j]

      index = random.randint(0, len(van2) - 1)
      populacao_crossover.append(ind1[0:index] + ind2[index:])
      populacao_crossover.append(ind2[0:index] + ind1[index:])

  return populacao_crossover

def selecao2(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness2, reverse=True)
  return nova_populacao[0:tamanho_populacao]

# escolhe os indivíduos mais aptos
def selecao_com_tragedia2(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness2, reverse=True)
  if (geracao % geracoes_tragedia == 0):
    tamanho_tragedia = math.ceil(tamanho_populacao*tx_tragedia)
    novos_individuos = [gerar_individuo2() for _ in range(0, tamanho_populacao - tamanho_tragedia)]
    return nova_populacao[0:tamanho_tragedia] + novos_individuos
  else:
    return nova_populacao[0:tamanho_populacao]

#FUNCOES VAN3

def gerar_individuo3():
  resultado = []
  for i in range(len(van3)):
    k = random.randint(0,16)
    j = random.randint(0,16)
    str1 = str("DP" + str(k))
    str2 = str("DP" + str(j))
    randomica = matrix[str1][str2]
    resultado.append(randomica)
    #print(resultado)
  return resultado

def fitness3(senha):
  score = 0
  for i in range(len(van3)):
    if van3[i] == senha[i]: score += 0.25
  return score

# retorna populuacao mutada com uma taxa
def mutacao3(populacao):
  populacao_escolhida = random.choices(populacao, k=math.ceil(tx_mutacao*len(populacao)))
  #print(populacao_escolhida)
  return [mutacao_flip3(individuo) for individuo in populacao_escolhida]

# flip de valor de gene de um gene aleatório
def mutacao_flip3(individuo):
  novo_individuo = list(individuo)
  index = random.randint(0, len(individuo) - 1)
  k = random.randint(0,16)
  j = random.randint(0,16)
  str1 = str("DP" + str(k))
  str2 = str("DP" + str(j))
  nova_matrix = matrix[str1][str2]
  novo_individuo[index] = nova_matrix # mutando gene
  #print(novo_individuo)
  return novo_individuo

def crossover3(populacao, geracao):
  funcao_decaimento_crossover = 1 #math.exp(-geracao/200)
  qtd = funcao_decaimento_crossover*tx_crossover*len(populacao)
  populacao_crossover = []
  populacao_escolhida = random.choices(populacao, k=math.ceil(qtd))
  [1, 2, 3, 4]
  for i in range(len(populacao_escolhida) - 1):
    for j in range(i+1, len(populacao_escolhida)):
      ind1 = populacao_escolhida[i]
      ind2 = populacao_escolhida[j]

      index = random.randint(0, len(van3) - 1)
      populacao_crossover.append(ind1[0:index] + ind2[index:])
      populacao_crossover.append(ind2[0:index] + ind1[index:])

  return populacao_crossover

def selecao3(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness3, reverse=True)
  return nova_populacao[0:tamanho_populacao]

# escolhe os indivíduos mais aptos
def selecao_com_tragedia3(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness3, reverse=True)
  if (geracao % geracoes_tragedia == 0):
    tamanho_tragedia = math.ceil(tamanho_populacao*tx_tragedia)
    novos_individuos = [gerar_individuo3() for _ in range(0, tamanho_populacao - tamanho_tragedia)]
    return nova_populacao[0:tamanho_tragedia] + novos_individuos
  else:
    return nova_populacao[0:tamanho_populacao]

#FUNCOES VAN4

def gerar_individuo4():
  resultado = []
  for i in range(len(van4)):
    k = random.randint(0,16)
    j = random.randint(0,16)
    str1 = str("DP" + str(k))
    str2 = str("DP" + str(j))
    randomica = matrix[str1][str2]
    resultado.append(randomica)
    #print(resultado)
  return resultado

def fitness4(senha):
  score = 0
  for i in range(len(van4)):
    if van4[i] == senha[i]: score += 0.25
  return score

# retorna populuacao mutada com uma taxa
def mutacao4(populacao):
  populacao_escolhida = random.choices(populacao, k=math.ceil(tx_mutacao*len(populacao)))
  #print(populacao_escolhida)
  return [mutacao_flip4(individuo) for individuo in populacao_escolhida]

# flip de valor de gene de um gene aleatório
def mutacao_flip4(individuo):
  novo_individuo = list(individuo)
  index = random.randint(0, len(individuo) - 1)
  k = random.randint(0,16)
  j = random.randint(0,16)
  str1 = str("DP" + str(k))
  str2 = str("DP" + str(j))
  nova_matrix = matrix[str1][str2]
  novo_individuo[index] = nova_matrix # mutando gene
  #print(novo_individuo)
  return novo_individuo

def crossover4(populacao, geracao):
  funcao_decaimento_crossover = 1 #math.exp(-geracao/200)
  qtd = funcao_decaimento_crossover*tx_crossover*len(populacao)
  populacao_crossover = []
  populacao_escolhida = random.choices(populacao, k=math.ceil(qtd))
  [1, 2, 3, 4]
  for i in range(len(populacao_escolhida) - 1):
    for j in range(i+1, len(populacao_escolhida)):
      ind1 = populacao_escolhida[i]
      ind2 = populacao_escolhida[j]

      index = random.randint(0, len(van4) - 1)
      populacao_crossover.append(ind1[0:index] + ind2[index:])
      populacao_crossover.append(ind2[0:index] + ind1[index:])

  return populacao_crossover

def selecao4(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness4, reverse=True)
  return nova_populacao[0:tamanho_populacao]

# escolhe os indivíduos mais aptos
def selecao_com_tragedia4(populacao, geracao):
  nova_populacao = sorted(populacao, key=fitness4, reverse=True)
  if (geracao % geracoes_tragedia == 0):
    tamanho_tragedia = math.ceil(tamanho_populacao*tx_tragedia)
    novos_individuos = [gerar_individuo4() for _ in range(0, tamanho_populacao - tamanho_tragedia)]
    return nova_populacao[0:tamanho_tragedia] + novos_individuos
  else:
    return nova_populacao[0:tamanho_populacao]



#main()

# EXECUÇÃO VAN 1 
populacao = []

for i in range(tamanho_populacao):
  populacao.append(gerar_individuo())

populacao = sorted(populacao, key=fitness, reverse=True)
geracao = 0

while fitness(populacao[0]) < 1:
  geracao += 1
  populacao_mutada = mutacao(populacao)
  populacao_crossover = crossover(populacao, geracao)
  if geracao <= 1000: 
    populacao = selecao(populacao_mutada + populacao + populacao_crossover, geracao)
  else:
    populacao = selecao_com_tragedia(populacao_mutada + populacao + populacao_crossover, geracao)

  """ if geracao % 100 == 0 or (geracao % 10 == 0 and geracao < 100):
      print("---------------- Intermediário: " + str(geracao)+ " ----------------")
      print("Van 1: " + str(populacao[0]))
      print("Tx Acerto: " + str(round(fitness(populacao[0]),2) * 100) + " %")
   """

print("---------------- Final " + str(geracao) + " ----------------")
print("Van 1: " + str(populacao[0]))
print("CAMINHO VAN 1 = " + str(dist_van1[0]) + " => " + str(dist_van1[1]) + " => " + str(dist_van1[2]) + " => " + str(dist_van1[3]) + " => " + str(dist_van1[4]) + " => " + str(dist_van1[5]))
print("Tx Acerto: " + str(round(fitness(populacao[0]),2)* 100) + " %")

# EXECUÇÃO VAN 2
populacao = []

for i in range(tamanho_populacao):
  populacao.append(gerar_individuo2())

populacao = sorted(populacao, key=fitness2, reverse=True)
geracao = 0

while fitness2(populacao[0]) < 1:
  geracao += 1
  populacao_mutada = mutacao2(populacao)
  populacao_crossover = crossover2(populacao, geracao)
  if geracao <= 1000: 
    populacao = selecao2(populacao_mutada + populacao + populacao_crossover, geracao)
  else:
    populacao = selecao_com_tragedia2(populacao_mutada + populacao + populacao_crossover, geracao)
  """ if geracao % 100 == 0 or (geracao % 10 == 0 and geracao < 100):
    print("---------------- Intermediário: " + str(geracao)+ " ----------------")
    print("Van 2: " + str(populacao[0]))
    print("Tx Acerto: " + str(round(fitness2(populacao[0]),2) * 100) + " %") """


print("---------------- Final " + str(geracao) + " ----------------")
print("Van 2: " + str(populacao[0]))
print("CAMINHO VAN 2 = " + str(dist_van2[0]) + " => " + str(dist_van2[1]) + " => " + str(dist_van2[2]) + " => " + str(dist_van2[3]) + " => " + str(dist_van2[4]) + " => " + str(dist_van2[5]))
print("Tx Acerto: " + str(round(fitness2(populacao[0]),2)* 100) + " %")

# EXECUÇÃO VAN 3 
populacao = []

for i in range(tamanho_populacao):
  populacao.append(gerar_individuo3())

populacao = sorted(populacao, key=fitness3, reverse=True)
geracao = 0

while fitness3(populacao[0]) < 1:
  geracao += 1
  populacao_mutada = mutacao3(populacao)
  populacao_crossover = crossover3(populacao, geracao)
  if geracao <= 1000: 
    populacao = selecao3(populacao_mutada + populacao + populacao_crossover, geracao)
  else:
    populacao = selecao_com_tragedia3(populacao_mutada + populacao + populacao_crossover, geracao)
  """ if geracao % 100 == 0 or (geracao % 10 == 0 and geracao < 100):
    print("---------------- Intermediário: " + str(geracao)+ " ----------------")
    print("Van 3: " + str(populacao[0]))
    print("Tx Acerto: " + str(round(fitness3(populacao[0]),2) * 100) + " %") """


print("---------------- Final " + str(geracao) + " ----------------")
print("Van 3: " + str(populacao[0]))
print("CAMINHO VAN 3 = " + str(dist_van3[0]) + " => " + str(dist_van3[1]) + " => " + str(dist_van3[2]) + " => " + str(dist_van3[3]) + " => " + str(dist_van3[4]) + " => " + str(dist_van3[5]))
print("Tx Acerto: " + str(round(fitness3(populacao[0]),2)* 100) + " %")

# EXECUÇÃO VAN 4
populacao = []

for i in range(tamanho_populacao):
  populacao.append(gerar_individuo4())

populacao = sorted(populacao, key=fitness4, reverse=True)
geracao = 0

while fitness4(populacao[0]) < 1:
  geracao += 1
  populacao_mutada = mutacao4(populacao)
  populacao_crossover = crossover4(populacao, geracao)
  if geracao <= 1000: 
    populacao = selecao4(populacao_mutada + populacao + populacao_crossover, geracao)
  else:
    populacao = selecao_com_tragedia4(populacao_mutada + populacao + populacao_crossover, geracao)
  """ if geracao % 100 == 0 or (geracao % 10 == 0 and geracao < 100):
    print("---------------- Intermediário: " + str(geracao)+ " ----------------")
    print("Van 4: " + str(populacao[0]))
    print("Tx Acerto: " + str(round(fitness4(populacao[0]),2) * 100) + " %")  """


print("---------------- Final " + str(geracao) + " ----------------")
print("Van 4: " + str(populacao[0]))
print("CAMINHO VAN 4 = " + str(dist_van4[0]) + " => " + str(dist_van4[1]) + " => " + str(dist_van4[2]) + " => " + str(dist_van4[3]) + " => " + str(dist_van4[4]) + " => " + str(dist_van4[5]))
print("Tx Acerto: " + str(round(fitness4(populacao[0]),2)* 100) + " %")