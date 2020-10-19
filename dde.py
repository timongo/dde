import numpy as np

class ddeint():
    def __init__(self,f,tau,y0):
        """
        Solver for the following dde
        dy/dt = f(t,y(t),y(t-tau))
        with initial condition
        y(t) = y0(t) for -tau < t < 0

        Method is RK4 with a time step that divides the delay tau
        
        For the interval 0 < t < tau, one solves the ode
        dy/dt = g1(t,y) with g(t,y) = f(t,y,y0(t-tau))
        this yields solution y1(t)

        For subsquent times, the time step is such that the quantities
        y(t-tau) are already known.
        """

        self.f = f
        self.tau  = tau
        self.y0 = y0
        if type(y0(0))==list or type(y0(0))==np.ndarray:
            self.N = len(y0(0))
        else:
            self.N = 1

    def rk4_step(self,dt,t,y,ytmtau,ytdt2mtau,ytdtmtau):
        """
        Runge-Kutta order 4
        """
        f = self.f

        k1 = f(t,y,ytmtau)
        k2 = f(t+0.5*dt,y+0.5*dt*k1,ytdt2mtau)
        k3 = f(t+0.5*dt,y+0.5*dt*k2,ytdt2mtau)
        k4 = f(t+dt,y+dt*k3,ytdtmtau)

        return y + dt*(k1+2.*k2+2.*k3+k4)/6.
        
    def Evolve(self,n,tmax):
        """
        n is an integer. The time step is tau/(2*n)
        We solve on the interval [0,tau*E[tmax/tau]] (E[] is the floor function)
        """

        tau = self.tau
        N = self.N
        L = int(tmax/tau)
        y0 = self.y0

        U = np.zeros((2*L*n+1,N))

        # For the case of 0<t<tau, no care needs to be taken
        dt = tau/(2.*float(n))
        U[0,:] = y0(0)
        t = 0.
        for i in range(1,2*n+1):
            ytmtau = y0(t-tau)
            ytdt2mtau = y0(t+0.5*dt-tau)
            ytdtmtau = y0(t+dt-tau)
            U[i,:] = self.rk4_step(dt,t,U[i-1,:],ytmtau,ytdt2mtau,ytdtmtau)
            t = t+dt
            
        # Now we ought to be careful. Knitting begins
        dt = tau/float(n) # time step is doubled
        # to get the first unknown quantity, which lives at t=tau+dt/2, we integrate starting
        # at t=tau-dt/2, because that's where we can have all the required information
        t = tau-dt/2.
        for i in range(2*n+1,2*L*n+1):
            if i>2*n+1:
                ytmtau=U[i-(2*n+2),:]
            else:
                # Special case where t-tau < 0
                ytmtau=y0(t-tau)
            ytdt2mtau=U[i-(2*n+1),:]
            ytdtmtau=U[i-2*n,:]
            U[i,:] = self.rk4_step(dt,t,U[i-2,:],ytmtau,ytdt2mtau,ytdtmtau)
            t = t+dt/2.

        # reconstruct the time array
        t = np.linspace(0,tau*L,2*L*n+1)
        return t,U
