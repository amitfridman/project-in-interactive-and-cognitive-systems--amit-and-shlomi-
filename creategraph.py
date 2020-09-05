import pickle
import numpy as np
from matplotlib import pyplot as plt 

lrp=pickle.load(open('scores_lrp_9x9_1.pkl','rb'))
sa=pickle.load(open('scores_sa_9x9_1.pkl','rb'))
x=np.arange(0,25,1)

plt.plot(x,sa,label='sa')
plt.plot(x,lrp,label='lrp')
plt.title('Top 5 Score by Number of Pertubations')
plt.xlabel('Number of pertubations')
plt.ylabel('Top 5 score')
plt.legend()
plt.savefig('lrp_vs_sa_pertubations_1.png')

