/* Block Processing GPU Proram */

/******************************************************
This is a simple software correlator for complex input data. The input is a stream of data samples (2 bytes)
with one group of bytes for each input per time sample. I.e. for 4 inputs, at time t,
the input would be:
i0i1i2i3 for time t0, then i0i1i2i3 for time t1 etc.
The output is a file of binary float data, real or complex, with adjacent values being the
spectral channels of the correlation products. I.e for channel c, product p the output would be
c0c1...cnchan-1 for p then
c0c1...cnchan-1 for p+1 etc.

Author: Randall Wayth, Feb 2008.

Requires:
- fftw3, including single precision libraries

to compile:
gcc -Wall -O3 -D_FILE_OFFSET_BITS=64 -o corr_cpu corr_cpu.c -lfftw3f -lm

O3 optimisation gives the fastest code, but is impossible to debug with gdb since
the compiler inlines everything.
Switch to -O -g optimisation or no optimisation for debugging.

Use:
start the program with no arguments for a summary of command line options.
Command-line options control the number of inputs, number of freq channels,
type of correlation product to write (auto, cross or both),
the number of averages before writing output and the input and output file names.
(stdin,stdout can also be used in some cases)
Debugging/timing output can also be generated with the command line option '-d'.

The code reads blocks of ninp*nchan complex samples and re-shuffles them into ninp arrays of nchan complex floats.
(for nchan spectral channels, not including the total power term, one needs nchan complex values to FFT.)
Each input array is then FFT'd into a complex spectrum. Correlation products (auto and/or cross)
are then generated for each of these spectra and accumulated.

After the specified number of spectra have been added into
the accumulations (the number of averages), they are written to output after being
averaged and normalised for the FFT. In this way, one or more averages can be written per output file.
If the end of the input is reached and nothing has yet been written, one average is written regardless
of the command-line '-a' argument. If one or more averages have already been written and the end of
the input has been reached, then partially accumulated data is discarded.

By default, two output files are created with the suffixes ".LACSPC" and ".LCCSPC" for the
auto and cross correlation products respectively. If only one type of correlation product is generated,
then stdout can be used.
The output format of autocorrelation is binary single precision floats,
one spectrum for each correlation product, per average.
The output is ordered by channel, then product, then average. I.e. for 4 output products with 512 channels,
the data will be channel1, channel2... channel512 all for product 1, then the same for
channel 2 etc. Then the whole thing is repeated again if there are more than 1 averages.
The output format of cross correlation is binary single precision complexes,
one spectrum for each correlation product, per average. Same idea with data ordering.

It is advisable to use input files that are an integer multiple of ninp*nchan samples long, otherwise there
will be unused data at the end of the file. If more than one average put input file is desired,
it is also advisable to use files that are ninp*nchan*naver samples long so that all data is used in an output average.

How many averages do I need to average for t seconds?
- for Nyquist sampled byte data with bandwidth B Hz, there are B*n_inputs samples per second.
- for n_channels output, we consume n_chan*n_input samples per input batch.
- therefore B/nchan (batches/sec) gives the number of averages required for 1 second of data.

Input data size:
this code was written for MWA 8T/32T data with 2 byte complex input samples. The input is 5real+5imag
bits in the lowest bits of the 16-bit word. The 5-bit numbers are two's complement integers.

******************************************************/

/*

ver0
Modified to process CARSE data. Based on Will Delligner's program for reading CARSE data.

ver1
Now can perform arbirary length FFT but the count values to make sure there is no packet drop is NOT checked

ver2
We switched to fftw3f

Anish Aug 14, 2024

*/


/*
        [To-do]

                -
 */

 /*
        [Compilation]
            [Efficient] nvcc -O3 block_gpu_complex.cu -lcufft -o block_gpu
            [Debugging] nvcc -g -G block_gpu_complex.cu -lcufft -o block_gpu
        
        [Execution]
            [Efficient]  ./block_gpu -p B -c 1024 -n 4 -i /mnt/carsedat/20250220_181537_0000.dat -o output -d
            [Debugging] gdb ./block_gpu   ------> set -p A -c 1024 -n 4 -i /mnt/carsedat/20250220_181537_0000.dat -o output -d    ---->  "run"

        
        Auto:   "A"
        Cross:  "C"
        Both:   "B"
            
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>
#include <complex.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define MAX_THREADS 1
#define BYTES_PER_SAMPLE 2
#define BITS_PER_NUM 5
#define TWOCOMP_MAXUINT   (1<<(BITS_PER_NUM))    // = 32 for 5 bits
#define TWOCOMP_ZEROPOINT (1<<(BITS_PER_NUM-1))  // = 16 for 5 bits
#define DECODER_MASK ((1<<(2*BITS_PER_NUM)) -1)
#define TESTING 1                                // 1 = testing (1 block), 0 otherwise
#define NUM_TESTING_BLOCKS 30
#define PKTS_PER_BLOCK (16 * 1024)
//#define PKTS_PER_BLOCK 1


#define CUFFT_CHECK(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT check output in file '%s' at line %d: %s\n", __FILE__, __LINE__, cufftGetErrorString(err)); \
    } \
}

const char* cufftGetErrorString(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "Unknown CUFFT error";
    }
}

typedef struct _fftplan {
    cufftHandle *plans;
    int doneplan;
} fftplan_t;

typedef struct _datareader {
    int init,ntoread, offset, nread; /* Added offset to skip the first 64 bytes or 32 int_16 values.
                                        Added nread; nread packets to be read to get the required channels Anish Aug 29 2024*/
    unsigned char *buffer;
    cufftComplex *decoder;
} datareader_t;

/* stores precomputed pairs depending on prod_type*/
typedef struct Pair{
    int inp1;
    int inp2;
    int idx;
}Pair;

typedef struct PairList{
    Pair* pairs;
    int count;
} PairList;

/* function prototypes */
void print_usage(char * const argv[]);
void parse_cmdline(int argc, char * const argv[], const char *optstring);
int readData(datareader_t *reader,int nchan,int ninp,FILE *fpin,cufftComplex *inp_buf);
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc);
void do_FFT_CUDA(fftplan_t *plan, int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf);
void do_CMAC(const int nchan,const int ninp,const int prod_type, cufftComplex *ft_buf, cufftComplex *corr_buf);
void writeOutput(FILE *fout_ac,FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,cufftComplex *buf, float normaliser, int ncorr);
float elapsed_time(struct timeval *start);
void dumpInputBuffer(const char *filename, cufftComplex *inp_buf, int ninp, int nchan);
__global__ void do_CMAC_CUDA(const int nchan,const int ninp, cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs);
PairList computePairs(int ninp, int prod_type, int ncorr);
void printPairList(PairList pList);
__global__ void do_CMAC_Block(const int nchan, const int ninp, const int packets_per_block, cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs);
__global__ void do_CMAC_Block_2(const int nchan, const int ninp, const int packets_per_block, cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs);

/* global vars */
int pktsize=8256; /* packet size of FPGA data in bytes - Anish Aug 26, 2024 */
int defnchan=1024;      /* Default number of channels in output spectra - Anish Aug 26, 2024 */
int nchan=1024;      /* number of channels in output spectra - Anish July 5, 2024 */
int ninp=4;         /* input of inputs */
int debug=0;        /* turn debugging info on */
int naver=99999;    /* number of input spectra to accumulate before writing an average */
int prod_type='B';  /* correlation product type: B: both, C: cross, A: auto */
char *infilename=NULL,*outfilename=NULL;
int nskip=0;        /* skip nchan * nskip data from the begining of the file  - set to 0 Anish July 5, 2024*/

#define SAMPLES_PER_BLOCK (PKTS_PER_BLOCK * ninp * nchan)   /* This will need to be changed later to support proper cmd line input */

int main(int argc, char * const argv[]) {
    int filedone=0,res=0,i,ncorr,iter=0,nav_written=0;
    FILE *finp=NULL,*fout_ac=NULL,*fout_cc=NULL;
    char optstring[]="dc:i:o:n:a:p:s:";
    float normaliser=1.0;
    float read_time=0, fft_time=0, cmac_time=0, write_time=0;
    cufftComplex *inp_buf=NULL, *ft_buf=NULL, *corr_buf=NULL;
    struct timeval thetime;
    fftplan_t fftplan[MAX_THREADS];
    datareader_t reader[MAX_THREADS];
    cudaStream_t stream;


    if(TESTING){
        naver = NUM_TESTING_BLOCKS;
    }

    /* initialise */
    for(i=0;i<MAX_THREADS;i++) {
        fftplan[i].doneplan=0;
        fftplan[i].plans=NULL;
        reader[i].init=0;
        reader[i].buffer=NULL;
        reader[i].decoder=NULL;
    }

    /* process command line args */
    if (argc <2) print_usage(argv);
    parse_cmdline(argc,argv,optstring);
    ncorr = ninp*(ninp+1)/2;
    if (debug) {
        fprintf(stderr,"Num inp:\t%d. Num corr products: %d\n",ninp,ncorr);
        fprintf(stderr,"Num chan:\t%d\n",nchan);
        fprintf(stderr,"Correlation product type:\t%c\n",prod_type);
        fprintf(stderr,"infile: \t%s\n",infilename);
        fprintf(stderr,"outfile:\t%s\n",outfilename);
    }

    /* open input and output files */
    openFiles(infilename,outfilename,prod_type, &finp,&fout_ac,&fout_cc);

    /* initialize CUDA */
    cudaFree(0);

    /* create CUDA stream for async operations */
    cudaStreamCreate(&stream);

    /* allocate buffers */
    /* for input and FFT results */
    cudaMallocHost((void**)&inp_buf, SAMPLES_PER_BLOCK * sizeof(cufftComplex));
    cudaMallocHost((void**)&ft_buf, SAMPLES_PER_BLOCK * sizeof(cufftComplex));

    /* initialize to zero */
    memset(inp_buf, 0, SAMPLES_PER_BLOCK * sizeof(cufftComplex));
    memset(ft_buf, 0, SAMPLES_PER_BLOCK * sizeof(cufftComplex));

    /* allocate device buffers for CUDA */
    cufftComplex *inp_buf_dev, *ft_buf_dev;
    cudaMalloc(&inp_buf_dev, SAMPLES_PER_BLOCK * sizeof(cufftComplex));
    cudaMalloc(&ft_buf_dev, SAMPLES_PER_BLOCK * sizeof(cufftComplex));

    /* create cuFFT plan (can be reused) */
    if (!fftplan->plans) {
        fftplan->plans = (cufftHandle *)malloc(sizeof(cufftHandle));
        CUFFT_CHECK(cufftPlan1d(fftplan->plans, nchan, CUFFT_C2C, PKTS_PER_BLOCK * ninp));
        fftplan->doneplan = 1;
    }

    if (inp_buf == NULL || ft_buf==NULL) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
    }
    
    /* for correlation products */
    cudaMallocHost((void**)&corr_buf, ncorr * nchan * sizeof(cufftComplex));
    memset(corr_buf, 0, ncorr * nchan * sizeof(cufftComplex));

    /* correlation products gpu buffer */       
    cufftComplex *corr_buf_dev;
    cudaMalloc(&corr_buf_dev, ncorr * nchan * sizeof(cufftComplex));
    
    /* Initialize GPU correlation buffer to zero */
    cudaMemset(corr_buf_dev, 0, ncorr * nchan * sizeof(cufftComplex));
    

    /* pair list gpu */
    PairList pList = computePairs(ninp, prod_type, ncorr);
    Pair* pList_dev;
    cudaMalloc(&pList_dev, ncorr * sizeof(Pair));
    cudaMemcpy(pList_dev, pList.pairs, ncorr * sizeof(Pair), cudaMemcpyHostToDevice);

    //int blocks = pList.count;
    //int threads = nchan;

    //int blocks_gpu = pList.count * PKTS_PER_BLOCK;
    int blocks_gpu = pList.count;
    int threads_gpu = nchan;
    

    /*skip the first nskip x nchan data
      This is introduced to skip the tone burst during correlation
      Anish, Dec 03, 2010  */

    for(i=0; i<nskip; i++) {
        res = readData(reader,nchan, ninp,finp,inp_buf);
    }

    /* process file */
    while (!filedone) {

        /* read time chunk into buffers */
        gettimeofday(&thetime,NULL);
        res = readData(reader,nchan, ninp,finp,inp_buf);
        
        /* dump input buffer if test run*/
        // if(TESTING){
        //     dumpInputBuffer("input_buffer_dump.bin", inp_buf, ninp, nchan);
        // }

        if (res !=0) {
            filedone=1;
        }
        read_time += elapsed_time(&thetime);

        if (!filedone) {

            /*  copy buffer from CPU to GPU for processing (async) */
            cudaMemcpyAsync(inp_buf_dev, inp_buf, SAMPLES_PER_BLOCK * sizeof(cufftComplex), 
                           cudaMemcpyHostToDevice, stream);

            /* do the FFT */
            gettimeofday(&thetime,NULL);
            
            /* set FFT to use stream */
            CUFFT_CHECK(cufftSetStream(*fftplan->plans, stream));
            do_FFT_CUDA(fftplan,nchan,ninp,inp_buf_dev,ft_buf_dev);
            
            /* synchronize to ensure FFT completes before timing */
            cudaStreamSynchronize(stream);

            fft_time += elapsed_time(&thetime);

            /* do the CMAC */
            gettimeofday(&thetime,NULL);

            //do_CMAC_CUDA<<<blocks, threads, 0, stream>>>(nchan, ninp, ft_buf_dev, corr_buf_dev, pList_dev);
            //do_CMAC_Block<<<blocks_gpu, threads_gpu, 0, stream>>>(nchan, ninp, PKTS_PER_BLOCK, ft_buf_dev, corr_buf_dev, pList_dev);
            do_CMAC_Block_2<<<blocks_gpu, threads_gpu, 0, stream>>>(nchan, ninp, PKTS_PER_BLOCK, ft_buf_dev, corr_buf_dev, pList_dev);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
            }
            
            cudaStreamSynchronize(stream);

            cmac_time += elapsed_time(&thetime);

        }

        /* write and average if it is time to */
        if ((filedone && nav_written==0) || ++iter == naver) {
            gettimeofday(&thetime,NULL);

            fprintf(stderr, "Normaliser calculated with iter = %d\n", iter);

            normaliser = 1.0/(iter*PKTS_PER_BLOCK);   /* Multiplied by PKTS_PER_BLOCK since iter now tracks num of blocks processed rather than packets (removed nchan from calculation Sept 9 2025)*/

            /* Transfer correlation results from GPU only when writing */
            cudaMemcpy(corr_buf, corr_buf_dev, ncorr * nchan * sizeof(cufftComplex), cudaMemcpyDeviceToHost);


            writeOutput(fout_ac,fout_cc,ninp,nchan,iter,prod_type,corr_buf,normaliser,ncorr);
            if(debug) fprintf(stderr,"writing average of %d chunks\n",iter);
            
            /* Reset GPU correlation buffer for next accumulation */
            cudaMemset(corr_buf_dev, 0, ncorr * nchan * sizeof(cufftComplex));
            
            iter=0;
            nav_written++;
            write_time += elapsed_time(&thetime);

            /* Stop running program early if in testing mode */
            if(TESTING) break;
        }

    }

    if (debug) {
        fprintf(stderr,"wrote %d averages. unused chunks: %d\n",nav_written,iter);
        fprintf(stderr,"Time reading:\t%g ms\n",read_time);
        fprintf(stderr,"Time FFTing:\t%g ms\n",fft_time);
        fprintf(stderr,"Time CMACing:\t%g ms\n",cmac_time);
        fprintf(stderr,"Time writing:\t%g ms\n",write_time);
    }

    /* clean up */
    fclose(finp);
    if(fout_ac !=NULL) fclose(fout_ac);
    if(fout_cc !=NULL) fclose(fout_cc);

    cufftDestroy(*fftplan->plans);
    cudaFreeHost(inp_buf); cudaFreeHost(ft_buf); cudaFreeHost(corr_buf); free(pList.pairs);
    cudaFree(inp_buf_dev); cudaFree(ft_buf_dev); cudaFree(corr_buf_dev); cudaFree(pList_dev);
    cudaStreamDestroy(stream);

    for(i=0;i<MAX_THREADS;i++) {
        if(reader[i].buffer) cudaFreeHost(reader[i].buffer);
        if(reader[i].decoder) free(reader[i].decoder);
    }

    return 0;
}


/* write out correlation products.
   Apply a normalisation factor that depends on the FFT length and the number
   of averages so the flux density is the same regardless of the spectral channel width */
void writeOutput(FILE *fout_ac, FILE *fout_cc,int ninp, int nchan, int naver, int prod_type,
                 cufftComplex *buf,float normaliser, int ncorr) {
    int inp1,inp2,cprod=0,chan,idx;
    float *temp_buffer=NULL;

    temp_buffer= (float*) malloc(sizeof(float)*(nchan));

    for(inp1=0; inp1<ninp; inp1++) {
        for (inp2=inp1; inp2<ninp; inp2++) {
            /* make an average by dividing by the number of chunks that went into the total */
            for (chan=0; chan<nchan; chan++) {

                idx = cprod * nchan + chan;

                buf[idx].x *= normaliser;
                buf[idx].y *= normaliser;

                /* convert the autocorrelation numbers into floats, since the imag parts will be zero*/
                if (inp1==inp2 && (prod_type == 'A' || prod_type=='B')){
                    temp_buffer[chan] = buf[idx].x;
                }
            }
            if(inp1==inp2 && (prod_type == 'A' || prod_type=='B')) {
                /* write the auto correlation product */
                fwrite(temp_buffer,sizeof(float),nchan,fout_ac);
            }
            if(inp1!=inp2 && (prod_type == 'C' || prod_type=='B')) {
                /* write the cross correlation product */
                fwrite(&buf[cprod * nchan],sizeof(cufftComplex),nchan,fout_cc);
            }

            /* reset the correlation products to zero */
            memset(&buf[cprod * nchan], '\0', nchan * sizeof(cufftComplex));
            cprod++;
        }
    }
    if (temp_buffer!=NULL) free(temp_buffer);
}


/* do the CMAC (cpu based) */
void do_CMAC(const int nchan,const int ninp,const int prod_type, cufftComplex *ft_buf, cufftComplex *corr_buf) {
    int inp1,inp2,cprod=0;
    cufftComplex ftinp1,ftinp2;
    register int chan;
    int idx;

    for(inp1=0; inp1<ninp; inp1++) {
        for(inp2=inp1; inp2<ninp; inp2++) {
            if(prod_type=='B' || ((prod_type=='C' && inp1!=inp2) || (prod_type=='A' && inp1==inp2))) {
                for(chan=0;chan<nchan;chan++) {                    
                    idx = cprod * nchan + chan;
                    ftinp1 = ft_buf[inp1 * nchan + chan];
                    ftinp2 = ft_buf[inp2 * nchan + chan];
                    cuFloatComplex prod = cuCmulf(ftinp1, cuConjf(ftinp2));
                    corr_buf[idx].x += prod.x;
                    corr_buf[idx].y += prod.y;
                }
            }
            cprod++;
        }
    }
}


/* do the FFT */
void do_FFT_CUDA(fftplan_t *fftplan, int nchan, int ninp, cufftComplex *inp_buf, cufftComplex *ft_buf) {
    /* execute */
    CUFFT_CHECK(cufftExecC2C(*fftplan->plans, inp_buf, ft_buf, CUFFT_FORWARD));
}


/* read in a batch of input data, n_channels*n_inputs complex samples.
   each sample has BITS_PER_NUM real and BITS_PER_NUM imaginary bits.
   convert to complex and pack into the input buffers for later FFT */
int readData(datareader_t *reader,int nchan,int ninp,FILE *fpin,cufftComplex *inp_buf) {

    int inp,chan,nread,ind, npkt;
//    unsigned short *sample; Anish
    int16_t *sample;
    int total_samples_read = 0;

    if (!reader->init) {
        int decoder_buf_size=(1<<(2*BITS_PER_NUM)),im,re,index=0,rval,ival;

        reader->init=1;
        reader->offset = 32;  /* 32 int_16 values to be skipped Anish July 5, 2024*/
        reader->nread = (int)(nchan/defnchan);  /* Number of reads to get the required channels Anish Aug 26, 2024*/
        reader->ntoread = pktsize; /* ininp*BYTES_PER_SAMPLE*nchan + reader->offset*BYTES_PER_SAMPLE;  /* number of bytes to read per batch including header Anish Aug 26,2024 */
        
        /* Allocate pinned buffer directly for faster transfers */
        cudaMallocHost((void**)&reader->buffer, reader->ntoread);
        reader->decoder = (cufftComplex *) malloc(sizeof(cufftComplex)*decoder_buf_size);

//The decoder is not needed for PAF
//Anish, Nov 9, 2010

        /* initialise the decoder lookup table for 5+5bit signed integer in twos complement format
        for(re=0;re< TWOCOMP_MAXUINT ;re++) {
            rval = re >= TWOCOMP_ZEROPOINT ? re-TWOCOMP_MAXUINT : re;
            for(im=0;im < TWOCOMP_MAXUINT;im++) {

                ival = im >= TWOCOMP_ZEROPOINT ? im-TWOCOMP_MAXUINT : im;

                reader->decoder[index] = rval + I*ival;
                //printf("decoder: (%g,%g)\n",creal(reader->decoder[index]),cimag(reader->decoder[index]));
                index++;
            }
        }*/

        if (debug) {
            fprintf(stderr,"size of read buffer: %d bytes\n",reader->ntoread);
            fprintf(stderr,"size of decoder lookup: %d elements\n",decoder_buf_size);
        }
    }

    for(int pkt = 0; pkt < PKTS_PER_BLOCK; pkt++){
        for(npkt=0; npkt<reader->nread; npkt++) {
            nread = fread(reader->buffer,1,reader->ntoread,fpin);
            if(nread < reader->ntoread) return 1;

            for(inp=0; inp<ninp; inp++) {
                for(chan=0; chan<nchan; chan++) {
                    //            sample = ((unsigned short *)reader->buffer)+(ninp*chan + inp); Anish
                    sample = ((int16_t*)reader->buffer)+(ninp*chan + inp + reader->offset);
                    //            ind = (*sample) & DECODER_MASK; Anish
                    //            inp_buf[inp][chan] = reader->decoder[ind]; Anish
                    //            inp_buf[inp][chan] = (*sample)/16 + I*0; Anish July 5, 2024
                    int idx = pkt * (ninp * nchan) + inp * nchan + chan;
                    //inp_buf[idx] = make_cuFloatComplex((float)(*sample)/16.0f, 0.0f);
                    inp_buf[idx] = make_cuFloatComplex((float)(*sample), 0.0f);     /* Removed divison by 4 Aug 15, 2025 */
                    //            printf("sample: %d. Val: (%g,%g)\n",(unsigned int) *sample, creal(inp_buf[inp][chan]),cimag(inp_buf[inp][chan]));
                    //            printf("sample: %d. Val: (%g,%g)\n",(int) *sample, creal(inp_buf[inp][chan]),cimag(inp_buf[inp][chan]));
                    total_samples_read++;
                }
            }
        }
    }

    // if(debug){
    //     fprintf(stderr, "Successfully read %d packets (%d samples)\n", PKTS_PER_BLOCK, total_samples_read);
    // }

    return 0;
}


/* open the input and output files */
int openFiles(char *infilename, char *outfilename, int prod_type, FILE **fin, FILE **fout_ac, FILE **fout_cc) {
    char tempfilename[FILENAME_MAX];

    if (infilename == NULL) {
        fprintf(stderr,"No input file specified\n");
        exit(1);
    }
    if (outfilename == NULL) {
        fprintf(stderr,"No output file specified\n");
        exit(1);
    }

    /* sanity check: can only use stdout for one type of output */
    if((prod_type=='B') && strcmp(outfilename,"-")==0) {
        fprintf(stderr,"Can only use stdout for either auto or cross correlations, not both\n");
        exit(1);
    }

    /* check for special file name: "-", which indicates to use stdin/stdout */
    if (strcmp(infilename,"-")==0) {
        *fin = stdin;
    } else {
        *fin = fopen(infilename,"r");
        if (*fin ==NULL) {
            fprintf(stderr,"failed to open input file name: <%s>\n",infilename);
            exit(1);
        }
    }

    if ((prod_type=='A') && strcmp(outfilename,"-")==0) {
        *fout_ac = stdout;
    } else if ((prod_type=='C') && strcmp(outfilename,"-")==0) {
        *fout_cc = stdout;
    } else {
        if (prod_type=='A' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LACSPC");
            *fout_ac = fopen(tempfilename,"w");
            if (*fout_ac ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        }
        if (prod_type=='C' || prod_type=='B') {
            strncpy(tempfilename,outfilename,FILENAME_MAX-8);
            strcat(tempfilename,".LCCSPC");
            *fout_cc = fopen(tempfilename,"w");
            if (*fout_cc ==NULL) {
                fprintf(stderr,"failed to open output file name: <%s>\n",tempfilename);
                exit(1);
            }
        }
    }

    return 0;
}


void parse_cmdline(int argc, char * const argv[], const char *optstring) {
    int c;

    while ((c=getopt(argc,argv,optstring)) != -1) {
        switch(c) {
            case 'c':
                nchan = atoi(optarg);
                if (nchan <=0 || nchan > 65536 || nchan % defnchan != 0) {  /* Added Anish Aug 26, 2024 */
                    fprintf(stderr,"bad number of channels: %d\n",nchan);
                    print_usage(argv);
                }
                break;
            case 'n':
                ninp = atoi(optarg);
                if (ninp <=0 || ninp > 128) {
                    fprintf(stderr,"bad number of inputs: %d\n",ninp);
                    print_usage(argv);
                }
                break;
            case 'a':
                naver = atoi(optarg);
                if (naver <=0 || naver > 9999999) {
                    fprintf(stderr,"bad number of averages: %d\n",naver);
                    print_usage(argv);
                }
                break;
            case 'p':
                prod_type = toupper(optarg[0]);
                if (prod_type!='A' && prod_type !='B' && prod_type != 'C') {
                    fprintf(stderr,"bad correlation product type: %c\n",prod_type);
                    print_usage(argv);
                }
                break;
            case 'i':
                infilename=optarg;
                break;
            case 'd':
                debug=1;
                break;
            case 'o':
                outfilename=optarg;
                break;
            case 's':
                nskip=atoi(optarg);
                break;
            default:
                fprintf(stderr,"unknown option %c\n",c);
                print_usage(argv);
        }
    }
}


void print_usage(char * const argv[]) {
    fprintf(stderr,"Usage:\n%s [options]\n",argv[0]);
    fprintf(stderr,"\t-p type\t\tspecify correlation product type(s). A: auto, C: cross, B: both. default: %c\n",prod_type);
    fprintf(stderr,"\t-c num\t\tspecify number of freq channels (should be multiples of default value). default: %d\n",nchan);  /* Anish Aug 26, 2024 */
    fprintf(stderr,"\t-n num\t\tspecify number of input streams. detault: %d\n",ninp);
    fprintf(stderr,"\t-a num\t\tspecify min number of averages before output. Default: %d\n",naver);
    fprintf(stderr,"\t-i filename\tinput file name. use '-' for stdin\n");
    fprintf(stderr,"\t-o filename\toutput file name. use '-' for stdout\n");
    fprintf(stderr,"\t-s num\tnum of nchan data to be skipped from the beginning of the data file\n"); //Anish Dec 03, 2010
    fprintf(stderr,"\t-d         \twrite debug and runtime info to stderr\n");
    exit(0);
}


/* returns the elapsed wall-clock time, in ms, since start (without resetting start) */
float elapsed_time(struct timeval *start){
    struct timeval now;
    gettimeofday(&now,NULL);
    return 1.e3f*(float)(now.tv_sec-start->tv_sec) +
           1.e-3f*(float)(now.tv_usec-start->tv_usec);
}


__global__ void do_CMAC_CUDA(const int nchan,const int ninp, cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs){

    int pair_idx = blockIdx.x;
    int chan = threadIdx.x;

    /* Bounds check */
    if (pair_idx >= gridDim.x || chan >= nchan) return;

    Pair pair = pairs[pair_idx];
    int inp1 = pair.inp1;
    int inp2 = pair.inp2;

    int idx1 = inp1 * nchan + chan;
    int idx2 = inp2 * nchan + chan;
    int idx_buf = pair.idx * nchan + chan;

    cufftComplex ftinp1 = ft_buf[idx1];
    cufftComplex ftinp2 = ft_buf[idx2];
    
    cufftComplex prod = cuCmulf(ftinp1, cuConjf(ftinp2));

    corr_buf[idx_buf].x += prod.x;
    corr_buf[idx_buf].y += prod.y;
}

/* precompute pairs for CMAC CUDA */
PairList computePairs(int ninp, int prod_type, int ncorr){

    PairList pList;
    int idx = 0;
    pList.count = 0;
    pList.pairs = (Pair*) malloc(ncorr * sizeof(Pair));

    if(!pList.pairs){
        fprintf(stderr, "Failed to allocate memory for pair list.\n");
        exit(EXIT_FAILURE);
    }
    
    int inp1, inp2;
    for(inp1=0; inp1<ninp; inp1++) {
        for(inp2=inp1; inp2<ninp; inp2++) {
            if(prod_type=='B' || ((prod_type=='C' && inp1!=inp2) || (prod_type=='A' && inp1==inp2))) {
                pList.pairs[pList.count].inp1 = inp1;
                pList.pairs[pList.count].inp2 = inp2;
                pList.pairs[pList.count].idx = idx;
                pList.count++;
            }
            idx++;
        }
    }

    return pList;
}

void printPairList(const PairList pList) {
    printf("PairList contains %d pairs:\n", pList.count);
    for (int i = 0; i < pList.count; ++i) {
        printf("  Pair %d: (inp1 = %d, inp2 = %d) and cprod = %d\n", i, pList.pairs[i].inp1, pList.pairs[i].inp2, pList.pairs[i].idx);
    }
}

__global__ void do_CMAC_Block(const int nchan, const int ninp, const int packets_per_block, cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs){

    int pair_idx = blockIdx.x / packets_per_block;
    int packet_idx = blockIdx.x % packets_per_block;
    int chan = threadIdx.x;

    /* Bounds Check */
    if(pair_idx >= gridDim.x / packets_per_block || chan >= nchan) return;

    Pair pair = pairs[pair_idx];
    int inp1 = pair.inp1;
    int inp2 = pair.inp2;

    /* Calculate indices for this specific packet in the block */
    int offset = packet_idx * ninp * nchan;
    int idx1 = offset + inp1 * nchan + chan;
    int idx2 = offset + inp2 * nchan + chan;
    int idx_buf = pair.idx * nchan + chan;

    
    cufftComplex ftinp1 = ft_buf[idx1];
    cufftComplex ftinp2 = ft_buf[idx2];

    cufftComplex prod = cuCmulf(ftinp1, cuConjf(ftinp2));

    atomicAdd(&corr_buf[idx_buf].x, prod.x);
    atomicAdd(&corr_buf[idx_buf].y, prod.y);

}

/* Perform CMAC on an entire block (16 * 1024 packets) */
__global__ void do_CMAC_Block_2(const int nchan, const int ninp, const int packets_per_block, 
                               cufftComplex *ft_buf, cufftComplex *corr_buf, Pair* pairs) {

    /* This code works by going through every possible pair/channel combination
     and performing computation on every packet for that pair/channel combination */
    
    int pair_idx = blockIdx.x;  /* Each block corresponds to an input pair */
    int chan = threadIdx.x;     /* Each thread corresponds to a channel (total of 10 * 1024 threads) */

    /* Bounds Check */
    if(pair_idx >= gridDim.x || chan >= nchan) return;

    /* Acquire indices for specific pair in this block */
    Pair pair = pairs[pair_idx];
    int inp1 = pair.inp1;
    int inp2 = pair.inp2;
    int idx_buf = pair.idx * nchan + chan;

    /* Accumulate across all packets in this block */
    cufftComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    
    /* Loop for every packet */
    for(int packet_idx = 0; packet_idx < packets_per_block; packet_idx++) {
        /* Calculate indices for this specific packet in the block */
        int offset = packet_idx * ninp * nchan;
        int idx1 = offset + inp1 * nchan + chan;
        int idx2 = offset + inp2 * nchan + chan;
        
        cufftComplex ftinp1 = ft_buf[idx1];
        cufftComplex ftinp2 = ft_buf[idx2];
        
        cufftComplex prod = cuCmulf(ftinp1, cuConjf(ftinp2));
        
        sum.x += prod.x;
        sum.y += prod.y;
    }
    
    /* Add the accumulated sum to the correlation buffer */
    atomicAdd(&corr_buf[idx_buf].x, sum.x);
    atomicAdd(&corr_buf[idx_buf].y, sum.y);
}



/* dumps input buffer to given file (used for testing)*/
// void dumpInputBuffer(const char *filename, cufftComplex *inp_buf, int ninp, int nchan){
//     FILE *f = fopen(filename, "wb");
//     if(!f){
//         perror("fopen for dumpInputBuffer");
//         return;
//     }

//     for(int i = 0; i < ninp; i++){
//         //assert(inp_buf[i] != NULL);
//         fwrite(inp_buf[i], sizeof(cufftComplex), nchan, f);
//     }

//     fclose(f);
// }
//     }

//     fclose(f);
// }
