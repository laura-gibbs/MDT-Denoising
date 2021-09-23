      program main
      
      implicit none

! !     Constants
! !---------------------------------------
!       integer,parameter :: rr=4.0d0              ! Resolution
! !---------------------------------------

!     Variable declarations
!---------------------------------------
      integer     :: i,j,II,JJ,rr               ! Lon/lat dimensions resolution
      real        :: frad                       ! Filter radius
      integer     :: frad_int
      character*3 :: fltr_type='gsn'            ! Filter type
      character(len=128)   :: name_mdt
      character(len=4)    :: frad_str
!---------------------------------------

      real,allocatable    :: msk(:,:)
      real,allocatable    :: mdt(:,:)
      real,allocatable    :: lon_d(:)
      real,allocatable    :: lat_d(:)

!     Output variables
!---------------------------------------
      real,allocatable     :: sdata(:,:)         ! Smoothed field
      real,allocatable     :: rsd(:,:)           ! The filter residual
!---------------------------------------



!     Required output grid
!-------------------------------------------------
      real*8 :: lon_stp       ! Longitude interval
      real*8 :: ltgd_stp      ! Latitude interval

      real*8 :: lon_min       ! Min longitude
      real*8 :: lon_max       ! Max longitude

      real*8 :: ltgd_min      ! Min latitude
      real*8 :: ltgd_max      ! Max latitude
!-------------------------------------------------

! ----------------------------------------
      logical :: msk_exists
      character(len=128) :: pin0,pin1,pout,fin,fin2,fout
!---------------------------------------

      pin0='C:/Users/oa18724/Documents/Master_PhD_folder/a_mdt_data/computations/masks/'
      pin1='C:/Users/oa18724/Documents/Master_PhD_folder/a_mdt_data/computations/mdts/mdts_tbf/'
      pout='C:/Users/oa18724/Documents/Master_PhD_folder/a_mdt_data/computations/mdts/gauss_filtered_mdts/'
!-------------------------------------------------
!===========================================================    

!     Read in the mdt parameter file
!--------------------------------------------------
      open(21,file='./filter_params.txt',form='formatted')
      read(21,'(A40)')name_mdt
      read(21,'(I4)')rr
      read(21,'(I6)')frad_int
      close(21)
      frad = REAL(frad_int)
      write(*,*) frad_int, frad


!-------------------------------------------------

!     Required output grid
!-------------------------------------------------
      lon_stp  = 1.0d0/rr              ! Longitude interval
      ltgd_stp = 1.0d0/rr              ! Latitude interval

      lon_min = 0.5d0*lon_stp            ! Min longitude
      lon_max = 360.0d0-0.5d0*lon_stp    ! Max longitude

      ltgd_min = -90.00d0+0.5d0*ltgd_stp   ! Min latitude
      ltgd_max =  90.00d0-0.5d0*ltgd_stp   ! Max latitude
!-------------------------------------------------

!     Compute grid dimensions
!     and make memory allocations 
!-------------------------------------------------
      II = nint((lon_max-lon_min)/lon_stp)+1
      JJ = nint((ltgd_max-ltgd_min)/ltgd_stp)+1
 
      allocate(mdt(II,JJ))
      allocate(msk(II,JJ))
      allocate(lon_d(II))
      allocate(lat_d(JJ))
      allocate(sdata(II,JJ))
      allocate(rsd(II,JJ))

!     Calculate longitude and geodetic latitude 
!     arrays for points on output grid
!-------------------------------------------------
      do i=1,II
         lon_d(i)=lon_stp*(i-1)+lon_min
      end do

      do j=1,JJ
         lat_d(j)=ltgd_stp*(j-1)+ltgd_min
      end do
!-------------------------------------------------


! ------------------------------------------------
!     Read in surfaces
!-------------------------------------------------
      fin=trim(name_mdt)//'.dat'
      open(20,file=trim(pin1)//trim(fin),&
      &form='unformatted')
      read(20)mdt
      close(20)

      fin2='mask_rr0004.dat'
      inquire(file=trim(pin0)//trim(fin2),exist=msk_exists)
      if(msk_exists)then
         open(20,file=trim(pin0)//trim(fin2),&
         &form='unformatted')
         read(20)msk
         close(20)
      else
         write(*,*)'mask does not exist'
         msk=0.0
      end if
!-------------------------------------------------
      write(frad_str, '(I4)')(frad_int/1000)

      call spatial_wmn_filter(II,JJ,lon_d,lat_d,mdt,msk,frad,fltr_type,sdata,rsd)
      fout=trim(name_mdt)//'_'//trim(adjustl(frad_str))//'k.dat'
      open(20,file=trim(pout)//trim(fout),&
      &form='unformatted')
      write(20)sdata
      close(20)

      deallocate(mdt,msk,lon_d,lat_d)
!-------------------------------------------------

      stop

!===========================================================  
      end program main


      subroutine spatial_wmn_filter(II,JJ,lon_d,lat_d,data,mask,frad,fltr_type,sdata,rsd)
!=====================================================================
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
!     Description:
!        Uses spatial averaging (weighted mean)to filter a geographical field.
!        The filter radius is supplied as an argument as is the filter type. 
!        A land mask, with land values non-zero, must be supplied on the same 
!        grid as the field to be filtered. Can use zero array if mask not important. 
!     
!     Created by:    
!        Rory Bingham
!
!     Created on:    
!        17/11/06
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!     Start of declarations
!-----------------------------------------------------------

      implicit none

!     Input variables
!---------------------------------------
      integer,intent(in)   :: II,JJ                ! Lon/lat dimensions
      real,intent(in)      :: lon_d(II),lat_d(JJ)  ! Lon/lats of grid points (degrees)
      real,intent(in)      :: data(II,JJ)          ! Field
      real,intent(in)      :: mask(II,JJ)          ! Land mask
      real,intent(in)      :: frad                 ! Filter radius 
      character*3,intent(in) :: fltr_type          ! Filter type
!---------------------------------------

!     Output variables
!---------------------------------------
      real,intent(out)     :: sdata(II,JJ)         ! Smoothed field
      real,intent(out)     :: rsd(II,JJ)           ! The filter residual
!---------------------------------------

!     Local variables
!---------------------------------------
      integer :: i,j,n,m,m1,m2,ltrd
      real    :: r,pi
      real    :: sigma
      real    :: x1,y1,z1,x2,y2,z2
      real    :: sd,ang,d
      real    :: fltr(II,JJ),tfltr(II,JJ),sm
      real    :: lon(II),lat(JJ)
!---------------------------------------

!-----------------------------------------------------------
!     End of declarations


!     Start of proceedure
!-----------------------------------------------------------

!     Initialise arrays
!---------------------------------------
      sdata(:,:)=0.0
!---------------------------------------

!     General parameters
!---------------------------------------
      r = 6378136.3      ! Earth radius
      pi=4.0*atan(1.0)
!---------------------------------------

!     Filter parameters
!---------------------------------------
      sigma = frad/sqrt(2.0*log(2.0))  ! sd of gaussian
!---------------------------------------

!     Convert lon/lat to radians
!---------------------------------------
      lon(:)=lon_d(:)*pi/180.0
      lat(:)=lat_d(:)*pi/180.0
!---------------------------------------
           
!     Filter
!---------------------------------------
      x1 = r*cos(lat(1))*cos(lon(1))
      y1 = r*cos(lat(1))*sin(lon(1))
      z1 = r*sin(lat(1))
      
   !  Compute merdional radius of filter
   !  (assume same for all latitudes)
   !---------------------------
      ltrd=1
      do
         x2 = r*cos(lat(ltrd))*cos(lon(1))
         y2 = r*cos(lat(ltrd))*sin(lon(1))
         z2 = r*sin(lat(ltrd))
         sd=sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
         ang = 2.0*asin(0.5*sd/r)
         d=r*ang
         if((fltr_type.eq.'box').and.(d.gt.frad))exit
         if((fltr_type.eq.'gsn').and.(d.gt.10.0*sigma))exit
         if((fltr_type.eq.'tgn').and.(d.gt.2*frad))exit
         if((fltr_type.eq.'han').and.(d.gt.2*frad))exit
         if((fltr_type.eq.'ham').and.(d.gt.2*frad))exit
         ltrd=ltrd+1
      end do
   !---------------------------

      do j=1,JJ

         write(*,*)'working on row',j

      ! Compute lat limits of filter window
      ! at latitude j
      !---------------------------
         m1=j-(ltrd-1)
         if(m1.lt.1)m1=1
         m2=j+(ltrd-1)
         if(m2.gt.JJ)m2=JJ
      !---------------------------

      ! Within above limits calculate 
      ! filter weights relative to first
      ! longitude
      !---------------------------
         fltr(:,:)=0.0
         tfltr(:,:)=0.0

         x1 = r*cos(lat(j))*cos(lon(1))
         y1 = r*cos(lat(j))*sin(lon(1))
         z1 = r*sin(lat(j))

         do m=m1,m2
            do n=1,II
               x2 = r*cos(lat(m))*cos(lon(n))
               y2 = r*cos(lat(m))*sin(lon(n))
               z2 = r*sin(lat(m))
               sd=sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
               ang = 2.0*asin(0.5*sd/r)
               d=r*ang
               if((fltr_type.eq.'box').and.(d.le.frad))then
                  fltr(n,m)=1.0
               end if
               if((fltr_type.eq.'gsn').and.(d.le.10.0*sigma))then
                  fltr(n,m)=exp(-0.5*d**2/sigma**2)/(sqrt(2*pi)*sigma)
               end if
               if((fltr_type.eq.'tgn').and.(d.le.2*frad))then
                  fltr(n,m)=exp(-0.5*d**2/sigma**2)/(sqrt(2*pi)*sigma)
               end if
               if((fltr_type.eq.'han').and.(d.le.2*frad))then
                  fltr(n,m)=0.54+0.46*cos(d*pi/(2*frad))
               end if
               if((fltr_type.eq.'ham').and.(d.le.2*frad))then
                  fltr(n,m)=0.5+0.5*cos(d*pi/(2*frad))
               end if
            end do
         end do
      !---------------------------

         do i=1,II

            if(mask(i,j).eq.0.0)then

            ! Translate the filter weights
            !---------------------------
               if(i.eq.1)then
                  tfltr(:,:)=fltr(:,:)
               else 
                  tfltr(1:i-1,:)=fltr(II-i+2:II,:)
                  tfltr(i:II,:)=fltr(1:II-i+1,:)
               end if 
         !---------------------------

            !  Apply filter
            !---------------------------
               sm=0.0
               do n=1,II
                  do m=m1,m2
                     if((tfltr(n,m).ne.0.0).and.(mask(n,m).eq.0.0))then
                        sdata(i,j)=sdata(i,j)+tfltr(n,m)*data(n,m)
                        sm=sm+tfltr(n,m)
                     end if
                  end do
               end do
               if(sm.ne.0.0)then
                  sdata(i,j)=sdata(i,j)/sm
               end if
               rsd(i,j)=data(i,j)-sdata(i,j)
            !---------------------------

            end if
 
         end do

      end do

!---------------------------------------

      return

!-----------------------------------------------------------
!     End of proceedure

!=====================================================================
      end subroutine spatial_wmn_filter



