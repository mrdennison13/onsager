import virial2 as virial
import eos
import numpy as np
import pandas as pd

class particle:


    B = {}
    iterations_tot = {}
    params = {}
    EOS = {}

    B_last = {}
    
    B_old = {}
    iterations_tot_old = {}

    n_vals = 1
    S2_out = np.array(0.0)
    cols = ['rho_i','rho_n','S2', 'Pr_i','Pr_n','mu_i','mu_n','fe_i','fe_n']
    inds= ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    coex = pd.DataFrame(index=inds, columns=cols)
    coex = coex.fillna(0)


    def __init__(self, shape = 'sphere', dim=(1.0)):
        import sys
        self.n_vals = 1
        self.S2_out = np.array(0.0)
        self.B = {}
        self.iterations_tot = {}
        self.params = {}
        self.EOS = {}
        self.B_last = {}
        self.B_old = {}
        self.iterations_tot_old = {}
        cols = ['rho_i','rho_n','S2', 'Pr_i','Pr_n','mu_i','mu_n','fe_i','fe_n']
        inds= ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        coex = pd.DataFrame(index=inds, columns=cols)
        self.coex = coex.fillna(0)

        dim = np.array(dim)
        i = 0
        for dd in dim:
            if dd < 0.0:
                print('Error: dim['+str(i)+'] is negative:- setting it to be positive')
                dim[i] = -dim[i]
                i+=1

        if shape == "sphere":
            if len(dim) != 1:
                print('Error: too many dimensions for sphere:- setting diamter to be dim[0]')
                sys.stdout.flush()
                tmp = dim[0]
                dim = np.empty(1)
                dim[0] = tmp
            self.vol = np.pi*dim[0]*dim[0]*dim[0]/6.0
                
        elif shape == "cut_sphere":
            if len(dim) == 1:
                print('Error: not enough dimensions for cut_sphere:- setting diamter to be 1')
                sys.stdout.flush()
                tmp = dim[0]
                dim = np.empty(2)
                dim[0] = tmp
                dim[1] = 1.0
            elif len(dim) > 2:
                print('Error: too many dimensions for cut_sphere:- using just first two')
                sys.stdout.flush()
                tmp = dim[0:2]
                dim = np.empty(2)
                dim = tmp
            if dim[0] > dim[1]:
                print('Error: dim[0] > dim[1]:- using dim[0]='+str(dim[1])+' and dim[1]='+str(dim[0]))
                sys.stdout.flush()
                tmp = dim[0]
                dim[0] = dim[1]
                dim[1] = tmp
            self.vol = np.pi*dim[0]*(3.0*dim[1]*dim[1]-dim[0]*dim[0])/12.0
                
        elif shape == "spherocyl":
            if len(dim) == 1:
                print('Error: not enough dimensions for spherocyl:- setting diamter to be 1')
                sys.stdout.flush()
                tmp = dim[0]
                dim = np.empty(2)
                dim[0] = tmp
                dim[1] = 1.0
            elif len(dim) > 2:
                print('Error: too many dimensions for spherocyl:- using just first two')
                sys.stdout.flush()
                tmp = dim[0:2]
                dim = np.empty(2)
                dim = tmp
            #if dim[1] > dim[0]:
            #    print('Error: dim[1] > dim[0]:- using dim[0]='+str(dim[1])+' and dim[1]='+str(dim[0]))
            #    sys.stdout.flush()
            #    tmp = [dim[1],dim[0]]
            #    dim = tmp
            self.vol = np.pi*dim[1]*dim[1]*dim[1]/6.0 + np.pi*dim[1]*dim[1]*dim[0]/4.0
                
        elif shape == "spheroid":
            if len(dim) == 1:
                print('Error: not enough dimensions for spheroid:- setting diamter to be 1')
                sys.stdout.flush()
                tmp = dim[0]
                dim = np.empty(2)
                dim[0] = tmp
                dim[1] = 1.0
            elif len(dim) > 2:
                print('Error: too many dimensions for spheroid:- using just first two')
                sys.stdout.flush()
                dim = dim[0:2]
            self.vol = np.pi*dim[0]*dim[1]*dim[1]/6.0
            
        elif shape == "oblate_spherocyl":
            if len(dim) == 1:
                print('Error: not enough dimensions for oblate_spherocyl:- setting diamter to be 1')
                sys.stdout.flush()
                tmp = dim[0]
                dim = np.empty(2)
                dim[0] = tmp
                dim[1] = 1.0
            elif len(dim) > 2:
                print('Error: too many dimensions for oblate_spherocyl::- using just first two')
                sys.stdout.flush()
                tmp = dim[0:2]
                dim = np.empty(2)
                dim = tmp
            if dim[0] > dim[1]:
                print('Error: dim[0] > dim[1]:- using dim[0]='+str(dim[1])+' and dim[1]='+str(dim[0]))
                sys.stdout.flush()
                tmp = dim[0]
                dim[0] = dim[1]
                dim[1] = tmp
            sigma = dim[1]-dim[0]
            self.vol = np.pi*dim[0]*dim[0]*dim[0]/6.0 + np.pi*np.pi*sigma*dim[0]*dim[0]/8.0 + np.pi*sigma*sigma*dim[0]/4.0
            
        self.dim = dim
        self.shape = shape
        print("Particle initialized as "+self.shape+" with dimensions:")
        i = 0
        for dd in dim:
            print("dim["+str(i)+"] = "+str(dd))
            i+=1
        print("Particle volume is: "+str(self.vol))
        return


    def restore_virial(self):
        import copy
        self.B = copy.deepcopy(self.B_old)  #self.B.copy()
        self.iterations_tot = copy.deepcopy(self.iterations_tot_old)  #self.iterations_tot.copy()
        return

    

    def calc_virial(self, orders=[2], iterations=[1000000], n_vals_in = 1, norm=True, timing=False):
        import sys
        import copy
        
        self.B_old = copy.deepcopy(self.B)  #self.B.copy()
        self.iterations_tot_old = copy.deepcopy(self.iterations_tot)  #self.iterations_tot.copy()

        if (2 not in orders) and ("B2" not in self.B.keys()) and (norm):
            return "Error: no data for B2:- set norm=False or add 2 to orders"

        if self.B:
            if (self.n_vals != n_vals_in) and (n_vals_in != 1):
                print('Error: can not change n_vals after initial run')
                print('n_vals is: ', self.n_vals)
                sys.stdout.flush()
        else:
            if (n_vals_in == "iso") or (n_vals_in == 0):
                self.n_vals = 1
                self.S2_out = np.linspace(0.0,0.0,self.n_vals)
            elif (n_vals_in == "nem") or (n_vals_in == 1):
                self.n_vals = 1
                self.S2_out = np.linspace(1.0,1.0,self.n_vals)
            elif type(n_vals_in) == int:
                self.n_vals = n_vals_in
                self.S2_out = np.linspace(0.0,1.0,self.n_vals)
            elif len(n_vals_in) > 1:
                self.S2_out = np.array(n_vals_in)
                self.n_vals = len(self.S2_out)
            else:
                return "Error: incorrect type for n_vals_in :"+str(type(n_vals_in))
            
        if timing:
            import time    

        nin = len(orders)
        nin_i = len(iterations)
        if (nin_i != nin) & (nin_i !=1):
            print('error: length of iterations > 1 but does not match length of orders:- setting all to iterations[0]')
            sys.stdout.flush()
            it1 = iterations[0]
            iterations = np.empty(nin)
            iterations[:] = it1
        elif nin_i == 1:
            it1 = iterations[0]
            iterations = np.empty(nin)
            iterations[:] = it1      
        for i in range(nin):
            order = orders[i]
            it1   = iterations[i]
            Bi = 'B'+str(order)            
            if order == 2:
                names = ('S2', Bi, 'error')
            else:
                names = ('S2', Bi+'*', 'error*')
            if time:
                t0 = time.clock()
            
            B_out = np.empty(self.n_vals)
            error_out = np.empty(self.n_vals)
            for i in range(self.n_vals):
                out = virial.calc_virial(order, self.shape, self.dim, it1, self.S2_out[i])
                B_out[i] = out[0]
                error_out[i] = out[1]
            z = np.array([self.S2_out,B_out,error_out])
            z = z.transpose()
            B_tmp = pd.DataFrame(z,columns=names)
            self.B_last[Bi] = B_tmp
            if Bi in self.B.keys():
                fac = self.iterations_tot[Bi]/(self.iterations_tot[Bi]+it1)
                self.B[Bi][names[1]] = fac*self.B[Bi][names[1]] + (1.0-fac)*B_tmp[names[1]]
                self.B[Bi][names[2]] = fac*np.sqrt(self.B[Bi][names[2]]**2 + (((1.0-fac)/fac)*B_tmp[names[2]])**2)
                self.iterations_tot[Bi]+=it1
            else:
                self.B[Bi]=B_tmp
                self.iterations_tot[Bi]=it1
            if time:
                t1 = time.clock()
                print('B'+str(order)+' complete: CPU time for '+str(it1)+' iterations:- ' +str(t1-t0)+'s')
                sys.stdout.flush()
            else:
                print('B'+str(order)+' complete')
                sys.stdout.flush()
            if norm:
                for Bi,Bn in self.B.items():
                    if (Bi != 'B2') & (Bi[0]=="B"):
                        i = int(Bn.columns[1][1])
                        Bii = Bi + '*'
                        self.B[Bi][Bi] = self.B[Bi][Bii]*(self.B['B2']['B2']**(i-1))
                        self.B[Bi]['error'] = np.sqrt((self.B[Bi]['error*']/self.B[Bi][Bii])**2 + (i-1.0)*(self.B['B2']['error']/self.B['B2']['B2'])**2)  *  self.B[Bi][Bi]    #(self.B['B2']['B2']**(i-1))    
        return


        

    def fit_func_virial(self, cut = False):
        from scipy.optimize import curve_fit
        for Bi, Bn in self.B.items():
            if (Bi[0]=="B"):
                if cut:
                    selection = (Bn['S2'] < 1.0)
                else:
                    selection = (Bn['S2'] < 1.1)
                S_section = Bn[selection]['S2']
                B_section = Bn[selection][Bi]
                error_section = Bn[selection]['error']
                B_max = B_section.max()
                tmp = curve_fit(eos.func, S_section, B_section,sigma=error_section)
                self.params[Bi] = tmp[0]
        return 



    def find_coex(self,rho_i=0.01, rho_n=4.0, order = 2, n_max = 1000, tol = 1e-10, updates=False):
        for i in range(2,order+1):
            Bi = "B"+str(i)
            if not Bi in self.params.keys():
                return "Error: function not fitted for "+Bi+": can't find coexistence"
        from scipy.optimize import minimize_scalar
        def diff_Pr():
            return  eos.pressure(alpha, rho_n, self.params, order) -  eos.pressure(0.0, rho_i, self.params, order)
        def diff_mu():
            return  eos.chemical_potential(alpha, rho_n, self.params, order) -  eos.chemical_potential(0.0, rho_i, self.params, order)
        for i in range(n_max):
            alpha = (minimize_scalar(eos.free_energy, args = (rho_n, self.params, order), bounds = (-1e-10,2e3), method='bounded')).x
            dPnn = eos.d_pr(alpha, rho_n, self.params, order)
            dPii = eos.d_pr(0.0, rho_i, self.params, order)
            dmnn = eos.d_mu(alpha, rho_n, self.params, order)
            dmii = eos.d_mu(0.0, rho_i, self.params, order)
            F1 = diff_Pr()
            F2 = diff_mu()
            dF11 = dPnn
            dF12 = -dPii
            dF21 = dmnn
            dF22 = -dmii
            det = dF22*dF11 - dF12*dF21
            if updates:
                print('loop: ',i)
                print('rho_i: ', rho_i, '  delta_i:', (F2*dF11 - F1*dF21)/det, '  Pr_i: ',eos.pressure(0.0, rho_i, self.params, order))
                print('rho_n: ', rho_n, '  delta_n:', (F1*dF22 - F2*dF12)/det, '  Pr_n: ',eos.pressure(alpha, rho_n, self.params, order))
                print('alpha: ', alpha)
                print('')
            rho_n1 = rho_n - (F1*dF22 - F2*dF12)/det
            rho_i1 = rho_i - (F2*dF11 - F1*dF21)/det
            if (rho_n1 < 0.0) or (rho_i1 < 0.0):
                return 'Error: try changing the guesses rho_i and rho_n'
            if i > 0:
                if(np.absolute(rho_n1 - rho_n)/rho_n < tol) & (np.absolute(rho_i1 - rho_i)/rho_i < tol):
                    coex = {}
                    coex['rho_i'] = rho_i1
                    coex['rho_n'] = rho_n1
                    coex['Pr_i'] = eos.pressure(0.0, rho_i1, self.params, order)
                    coex['Pr_n'] = eos.pressure(alpha, rho_n1, self.params, order)
                    coex['mu_i'] = eos.chemical_potential(0.0, rho_i1, self.params, order)
                    coex['mu_n'] = eos.chemical_potential(alpha, rho_n1, self.params, order)
                    coex['fe_i'] = eos.free_energy(0.0, rho_i1, self.params, order)
                    coex['fe_n'] = eos.free_energy(alpha, rho_n1, self.params, order)
                    coex['S2'] = eos.nematic_order(alpha)
                    cols = ['rho_i','rho_n','S2', 'Pr_i','Pr_n','mu_i','mu_n','fe_i','fe_n']
                    coex = pd.DataFrame(coex, index=np.arange(1))[cols] # pd.DataFrame(coex, index=['B'+str(order)])[cols]
                    self.coex.loc['B'+str(order),:] = coex.iloc[0,:] #pd.DataFrame(coex, index=np.arange(1))[cols]
                    return
            rho_n = rho_n1
            rho_i = rho_i1
        return 'Did not converge: try increasing n_max'



    def plot_virial(self,orders=[2]):
        import matplotlib.pyplot as plt
        ncols = 2
        nrows = len(orders)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols,3.5*nrows))
        ir = 0
        for order in orders:
            Bi = "B"+str(order)
            if Bi not in self.B.keys():
                plt.show()
                return "Error: no data for "+Bi
            x_min = 0.0
            x_max = 0.25 
            y_max1 = self.B[Bi][self.B[Bi]['S2'] <= 0.25][Bi].max()
            y_min1 = self.B[Bi][self.B[Bi]['S2'] <= 0.25][Bi].min()
            dy = y_max1-y_min1
            y_max1 += dy*0.2
            y_min1 -= dy*0.2
            if Bi in self.params.keys():
                x = np.linspace(0.0,1.0,100)
                y=eos.func(x,self.params[Bi][0],self.params[Bi][1],self.params[Bi][2],self.params[Bi][3],self.params[Bi][4],self.params[Bi][5],self.params[Bi][6])
            if nrows > 1:
                axs[ir,0].errorbar(self.B[Bi]['S2'],self.B[Bi][Bi],self.B[Bi]['error'], capthick=3, markersize=7,linestyle=None,marker="o",color="red", alpha = 0.7)
                if Bi in self.params.keys():
                    axs[ir,0].plot(x, y, color = "black", label = "fit", marker=None, linewidth=2.0)
                axs[ir,0].set_xlabel('S_nem')
                axs[ir,0].set_ylabel(Bi)
                axs[ir,0].set_title(Bi +' against nematic order parameter')
                axs[ir,0].margins(0.02)
                    
                axs[ir,1].errorbar(self.B[Bi]['S2'],self.B[Bi][Bi],self.B[Bi]['error'], capthick=3, markersize=7,linestyle=None,marker="o",color="red", alpha = 0.7)
                if Bi in self.params.keys():
                    axs[ir,1].plot(x, y, color = "black", label = "fit", marker=None, linewidth=2.0)
                axs[ir,1].set_xlabel('S_nem')
                axs[ir,1].set_title(Bi +' against nematic order parameter')
                axs[ir,1].set_xlim([x_min-0.02, x_max+0.02])
                axs[ir,1].set_ylim([y_min1, y_max1])
                axs[ir,1].margins(0.02)
                axs[ir,1].legend(loc='upper left', title="", bbox_to_anchor=(1.1, 1.0));
                ir +=1
            else:
                axs[0].errorbar(self.B[Bi]['S2'],self.B[Bi][Bi],self.B[Bi]['error'], capthick=3, markersize=7,linestyle=None,marker="o",color="red", alpha = 0.7)
                if Bi in self.params.keys():
                    axs[0].plot(x, y, color = "black", label = "fit", marker=None, linewidth=2.0)
                axs[0].set_xlabel('S_nem')
                axs[0].set_ylabel(Bi)
                axs[0].set_title(Bi +' against nematic order parameter')
                axs[0].margins(0.02)
            
                axs[1].errorbar(self.B[Bi]['S2'],self.B[Bi][Bi],self.B[Bi]['error'], capthick=3, markersize=7,linestyle=None,marker="o",color="red", alpha = 0.7)
                if Bi in self.params.keys():    
                    axs[1].plot(x, y, color = "black", label = "fit", marker=None, linewidth=2.0)
                axs[1].set_xlabel('S_nem')
                axs[1].set_title(Bi +' against nematic order parameter')
                axs[1].set_xlim([x_min-0.02, x_max+0.02])  
                axs[1].set_ylim([y_min1, y_max1])
                axs[1].margins(0.02)
                axs[1].legend(loc='upper left', title="", bbox_to_anchor=(1.1, 1.0))
        plt.tight_layout()
        plt.show()
  


    def calc_EOS(self,rho_min=0.001, rho_max=1.0, order = 2, n_rho = 1000):
        for i in range(2,order+1):
            Bi = "B"+str(i)
            if not Bi in self.params.keys():
                return "Error: function not fitted for "+Bi+": can't calculate the equation of state"
        rho_max /= self.vol
        Bi = "B"+str(order)
        self.EOS[Bi] = eos.calc_EOS(self.params, rho_min=rho_min, rho_max=rho_max, n_points=n_rho, order=order)
        return









    def plot_EOS(self, rho_min=0.001, rho_max=0, order = 2, n_rho = 1000):
        for i in range(2,order+1):
            Bi = "B"+str(i)
            if not Bi in self.params.keys():
                return "Error: function not fitted for "+Bi+": can't calculate the equation of state"
        import matplotlib.pyplot as plt
        if rho_max == 0:
            rho_max = 1.0/self.vol
        Bi = "B"+str(order)
        tmp = eos.calc_EOS(self.params, rho_min=rho_min, rho_max=rho_max, n_points=n_rho, order=order)
        plt.plot(tmp.rho,tmp.Pr_i, color = "green", label = "isotropic", marker=None, linewidth=2.0)
        plt.plot(tmp.rho,tmp.Pr_n, color = "red", label = "nematic", marker=None, linewidth=2.0)
        if self.coex.loc[Bi].S2 != 0:
            x = np.empty(2)
            y = np.empty(2)
            x[0] = self.coex.loc[Bi].rho_i
            x[1] = self.coex.loc[Bi].rho_n
            y[0] = self.coex.loc[Bi].Pr_i
            y[1] = self.coex.loc[Bi].Pr_n
            plt.plot(x,y, color = "black", label = "coexistence", marker='o', linewidth=2.0)
        plt.margins(0.02)
        plt.legend(loc='upper left', title="", bbox_to_anchor=(1.1, 1.0))
        plt.xlabel("density")
        plt.ylabel("pressure")
        plt.title("Equation of state at "+Bi+ " level")
        plt.show()
        return




    def output_virial(self, order=2, filename="B2-test.dat"):
        Bi = "B"+str(order)
        if Bi in self.B.keys():
            with open(filename, 'w') as fo:
                fo.write(self.B[Bi].to_string())
        else:
            return "Error: No data for "+Bi

            
    def output_EOS(self, order=2, filename="EOS_B2-test.dat"):
        Bi = "B"+str(order)
        if Bi in self.EOS.keys():
            with open(filename, 'w') as fo:
                fo.write(self.EOS[Bi].to_string())
        else:
            return "Error: No data for "+Bi


    def output_coex(self, filename="coex-test.dat"):
        if not self.coex[self.coex.S2 != 0].empty:
            with open(filename, 'w') as fo:
                fo.write(self.coex[self.coex.S2 != 0].to_string())
        else:
            return "Error: no coexistence data"
