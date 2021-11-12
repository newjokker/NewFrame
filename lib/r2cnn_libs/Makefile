all:
	cd box_utils/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd box_utils/cython_utils/; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd box_utils/cython_utils/; rm -rf */*.pyc rm -rf */*.so 
####### rm *.so *.c *.cpp; cd ../../
        


	#rm -rf */*.pyc
	#rm -rf */*.so
